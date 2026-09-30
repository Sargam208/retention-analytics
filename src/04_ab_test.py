"""
Step 4 - Retention A/B test (Cookie Cats, 90,189 players).

Experiment: the first "gate" (forced wait / pay) moved from level 30 (control) to 40.
Metrics: 1-day and 7-day retention (binary), plus game rounds played (guardrail).

Checklist implemented:
  1. Sample Ratio Mismatch (chi-square)            -> is the randomisation trustworthy?
  2. Two-proportion z-test + Wald CI               -> classic frequentist answer
  3. Bootstrap CI + bootstrap p-value              -> distribution-free check
  4. Holm correction across the two retention metrics
  5. Power analysis: MDE at this sample size, and n needed for the observed effect
  6. Guardrail on engagement (heavy-tailed -> Mann-Whitney + bootstrap of the median)
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep, proportion_effectsize, proportions_ztest,
)

from config import (
    AB_ALPHA, AB_CONTROL, AB_METRICS, AB_POWER, AB_TREATMENT, COOKIE_CATS_CSV, FIGURES,
    N_BOOTSTRAP, OUTPUTS, RANDOM_STATE,
)
from utils import PALETTE, log, save_json, setup_matplotlib

rng = np.random.default_rng(RANDOM_STATE)


def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def srm_check(n_c, n_t):
    chi2, p = stats.chisquare([n_c, n_t])          # expected 50/50
    return {"n_control": n_c, "n_treatment": n_t, "share_treatment": n_t / (n_c + n_t),
            "chi2": chi2, "p_value": p, "srm_detected": p < 0.001}


def bootstrap_diff_proportions(x_c, n_c, x_t, n_t, B=N_BOOTSTRAP):
    """Bootstrap of p_t - p_c.

    Resampling n Bernoulli outcomes with replacement and counting successes is exactly
    a Binomial(n, p_hat) draw, so this is identical to row-level resampling but
    ~10,000x faster.
    """
    pc = rng.binomial(n_c, x_c / n_c, size=B) / n_c
    pt = rng.binomial(n_t, x_t / n_t, size=B) / n_t
    return pt - pc


def mde_absolute(p_base, n_c, n_t, alpha=AB_ALPHA, power=AB_POWER):
    """Smallest absolute change detectable with this sample (two-sided)."""
    h = NormalIndPower().solve_power(effect_size=None, nobs1=n_c, alpha=alpha,
                                     power=power, ratio=n_t / n_c,
                                     alternative="two-sided")
    # Cohen's h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1))  ->  solve for p2
    p2 = np.sin(np.arcsin(np.sqrt(p_base)) + h / 2) ** 2
    return float(p2 - p_base)


def analyse_metric(df, metric):
    c = df.loc[df.version == AB_CONTROL, metric]
    t = df.loc[df.version == AB_TREATMENT, metric]
    n_c, n_t, x_c, x_t = len(c), len(t), int(c.sum()), int(t.sum())
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c

    z, p_z = proportions_ztest([x_t, x_c], [n_t, n_c], alternative="two-sided")
    lo, hi = confint_proportions_2indep(x_t, n_t, x_c, n_c, method="wald",
                                        compare="diff", alpha=AB_ALPHA)
    boot = bootstrap_diff_proportions(x_c, n_c, x_t, n_t)
    b_lo, b_hi = np.percentile(boot, [100 * AB_ALPHA / 2, 100 * (1 - AB_ALPHA / 2)])
    # two-sided bootstrap p-value: how often the resampled diff falls on the other side of 0
    p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))

    h_obs = proportion_effectsize(p_t, p_c)
    n_needed = (NormalIndPower().solve_power(effect_size=abs(h_obs), nobs1=None,
                                             alpha=AB_ALPHA, power=AB_POWER, ratio=1.0)
                if abs(h_obs) > 1e-9 else float("inf"))
    return {
        "metric": metric,
        "control_rate": p_c, "treatment_rate": p_t,
        "abs_diff": diff, "rel_lift": diff / p_c,
        "z_stat": float(z), "p_value_z": float(p_z),
        "ci_wald_low": float(lo), "ci_wald_high": float(hi),
        "ci_boot_low": float(b_lo), "ci_boot_high": float(b_hi),
        "p_value_bootstrap": p_boot,
        "prob_treatment_worse": float((boot < 0).mean()),
        "cohens_h": float(h_obs),
        "mde_abs_at_80_power": mde_absolute(p_c, n_c, n_t),
        "n_per_group_needed_for_observed_effect": float(np.ceil(n_needed)),
    }, boot


def engagement_guardrail(df):
    c = df.loc[df.version == AB_CONTROL, "sum_gamerounds"].values
    t = df.loc[df.version == AB_TREATMENT, "sum_gamerounds"].values
    u, p = stats.mannwhitneyu(t, c, alternative="two-sided")
    B = 2000
    med = np.array([np.median(rng.choice(t, len(t))) - np.median(rng.choice(c, len(c)))
                    for _ in range(B)])
    return {"median_control": float(np.median(c)), "median_treatment": float(np.median(t)),
            "mean_control": float(c.mean()), "mean_treatment": float(t.mean()),
            "mannwhitney_p": float(p),
            "median_diff_ci": [float(x) for x in np.percentile(med, [2.5, 97.5])]}


def main():
    plt = setup_matplotlib()
    df = pd.read_csv(COOKIE_CATS_CSV)
    for m in AB_METRICS:
        df[m] = to_bool(df[m]).astype(int)
    df = df[df.version.isin([AB_CONTROL, AB_TREATMENT])]
    log(f"{len(df):,} users | groups: {df.version.value_counts().to_dict()}")

    # data quality: one player famously logged ~50k rounds in 7 days
    q999 = df.sum_gamerounds.quantile(0.999)
    outliers = int((df.sum_gamerounds > 10 * q999).sum())
    never_played = float((df.sum_gamerounds == 0).mean())

    n_c = int((df.version == AB_CONTROL).sum())
    n_t = int((df.version == AB_TREATMENT).sum())
    srm = srm_check(n_c, n_t)
    log(f"SRM check p={srm['p_value']:.3f} -> {'PROBLEM' if srm['srm_detected'] else 'ok'}")

    rows, boots = [], {}
    for m in AB_METRICS:
        r, b = analyse_metric(df, m)
        rows.append(r); boots[m] = b
    res = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(res.p_value_z, alpha=AB_ALPHA, method="holm")
    res["p_value_holm"] = p_holm
    res["significant_after_holm"] = reject
    log(f"Results:\n{res[['metric', 'control_rate', 'treatment_rate', 'abs_diff', 'p_value_z', 'ci_boot_low', 'ci_boot_high', 'p_value_holm']].round(4).to_string(index=False)}")

    guard = engagement_guardrail(df)

    r7 = res.set_index("metric").loc["retention_7"] if "retention_7" in res.metric.values \
        else res.iloc[-1]
    if r7["significant_after_holm"] and r7["abs_diff"] < 0:
        decision = (f"Keep the gate at level 30. Moving it to 40 lowers 7-day retention by "
                    f"{abs(r7['abs_diff']) * 100:.2f} pp ({r7['rel_lift']:.1%} relative).")
    elif r7["significant_after_holm"] and r7["abs_diff"] > 0:
        decision = "Ship gate_40: it significantly improves 7-day retention."
    else:
        decision = "No significant difference in 7-day retention; keep the current gate."

    res.to_csv(OUTPUTS / "ab_results.csv", index=False)
    pd.DataFrame(boots).to_csv(OUTPUTS / "ab_bootstrap_samples.csv", index=False)
    rates = (df.groupby("version")[AB_METRICS].mean().reset_index()
             .melt(id_vars="version", var_name="metric", value_name="rate"))
    rates.to_csv(OUTPUTS / "ab_rates.csv", index=False)
    save_json({
        "n_users": len(df), "srm": srm,
        "data_quality": {"extreme_outliers": outliers, "share_never_played": never_played},
        "results": res.to_dict(orient="records"),
        "engagement_guardrail": guard,
        "decision": decision,
        "alpha": AB_ALPHA, "power": AB_POWER, "n_bootstrap": N_BOOTSTRAP,
    }, OUTPUTS / "ab_summary.json")
    log(decision)

    # ---- figures
    fig, axes = plt.subplots(1, len(AB_METRICS), figsize=(5 * len(AB_METRICS), 3.6))
    for ax, m in zip(np.atleast_1d(axes), AB_METRICS):
        ax.hist(boots[m] * 100, bins=60, color=PALETTE[0], alpha=0.85)
        ax.axvline(0, color="black", lw=1)
        r = res.set_index("metric").loc[m]
        for v in (r.ci_boot_low, r.ci_boot_high):
            ax.axvline(v * 100, ls="--", color=PALETTE[3])
        ax.set(title=f"{m}: bootstrap of {AB_TREATMENT} - {AB_CONTROL}",
               xlabel="difference (percentage points)")
    fig.tight_layout(); fig.savefig(FIGURES / "ab_bootstrap.png"); plt.close(fig)


if __name__ == "__main__":
    main()
