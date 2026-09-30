"""
Step 5 - Uplift modelling: who should receive the retention e-mail?

Data: Hillstrom / MineThatData (64,000 customers, randomised: Mens e-mail / Womens e-mail /
no e-mail; outcomes tracked for 2 weeks). All features are measured BEFORE the campaign.

A churn model answers "who is likely to leave?". Uplift answers "whose behaviour does the
campaign actually change?" - it separates Persuadables from Sure Things, Lost Causes and
Sleeping Dogs, which is what decides campaign ROI.

Models (all XGBoost based):
  S-learner          one model with treatment as a feature;  u(x) = f(x,1) - f(x,0)
  T-learner          separate models for treated / control;  u(x) = f1(x) - f0(x)
  Transformed outcome Y* = Y (T - e) / (e (1 - e)) with E[Y*|x] = u(x)  (Athey & Imbens)
Evaluation on a held-out 30%: Qini curve + coefficient, uplift@k, uplift by decile.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from config import (
    FIGURES, HILLSTROM_CSV, OUTPUTS, RANDOM_STATE, UPLIFT_TARGET, UPLIFT_TEST_SIZE,
    UPLIFT_TOP_K, UPLIFT_TREATMENT,
)
from utils import PALETTE, log, save_json, setup_matplotlib

XGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=50,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                  random_state=RANDOM_STATE)


# ----------------------------------------------------------------------------- data
def load():
    if HILLSTROM_CSV.exists():
        df = pd.read_csv(HILLSTROM_CSV)
    else:
        try:  # scikit-uplift downloads + caches the file
            from sklift.datasets import fetch_hillstrom
            b = fetch_hillstrom(target_col="all")
            df = pd.concat([b.data, b.target, b.treatment], axis=1)
            df.to_csv(HILLSTROM_CSV, index=False)
        except Exception as e:
            raise FileNotFoundError(f"Put hillstrom.csv in {HILLSTROM_CSV.parent} "
                                    f"(see data/README.md). ({e})")
    df.columns = [c.lower() for c in df.columns]
    return df


def prepare(df):
    if UPLIFT_TREATMENT == "any":
        t = (df.segment != "No E-Mail").astype(int)
    else:
        df = df[df.segment.isin([UPLIFT_TREATMENT, "No E-Mail"])].copy()
        t = (df.segment == UPLIFT_TREATMENT).astype(int)
    X = pd.get_dummies(
        df[["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]],
        columns=["zip_code", "channel"], dtype=int)
    X["history"] = np.log1p(X["history"])
    return X, t.values, df[UPLIFT_TARGET].values.astype(int), df


# ----------------------------------------------------------------------------- learners
def s_learner(Xtr, ttr, ytr, Xte):
    m = XGBClassifier(**XGB_PARAMS).fit(Xtr.assign(treatment=ttr), ytr)
    return (m.predict_proba(Xte.assign(treatment=1))[:, 1]
            - m.predict_proba(Xte.assign(treatment=0))[:, 1])


def t_learner(Xtr, ttr, ytr, Xte):
    m1 = XGBClassifier(**XGB_PARAMS).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = XGBClassifier(**XGB_PARAMS).fit(Xtr[ttr == 0], ytr[ttr == 0])
    return m1.predict_proba(Xte)[:, 1] - m0.predict_proba(Xte)[:, 1]


def transformed_outcome(Xtr, ttr, ytr, Xte):
    e = ttr.mean()                                 # known propensity (randomised test)
    y_star = ytr * (ttr - e) / (e * (1 - e))
    m = XGBRegressor(**XGB_PARAMS).fit(Xtr, y_star)
    return m.predict(Xte)


# ----------------------------------------------------------------------------- metrics
def qini_curve(y, t, score):
    """Incremental outcomes if we treat the top-x fraction ranked by `score`."""
    order = np.argsort(-score, kind="mergesort")
    y, t = y[order], t[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    frac = np.arange(1, len(y) + 1) / len(y)
    return np.r_[0, frac], np.r_[0, q]


def qini_coefficient(frac, q, n_treated):
    """Area between model curve and random targeting, per treated customer."""
    random = frac * q[-1]
    return float(np.trapezoid(q - random, frac) / n_treated) if hasattr(np, "trapezoid") \
        else float(np.trapz(q - random, frac) / n_treated)


def uplift_at_k(y, t, score, k):
    top = np.argsort(-score)[: int(k * len(y))]
    yt, tt = y[top], t[top]
    return float(yt[tt == 1].mean() - yt[tt == 0].mean())


def uplift_by_decile(y, t, score, n_bins=10):
    d = pd.DataFrame({"y": y, "t": t, "s": score}).sort_values("s", ascending=False)
    d["decile"] = np.arange(len(d)) * n_bins // len(d) + 1
    out = pd.DataFrame({
        "predicted_uplift": d.groupby("decile").s.mean(),
        "treated_rate": d[d.t == 1].groupby("decile").y.mean(),
        "control_rate": d[d.t == 0].groupby("decile").y.mean(),
        "n": d.groupby("decile").size(),
    })
    out["actual_uplift"] = out.treated_rate - out.control_rate
    return out.reset_index()


def ate_with_ci(y, t):
    p1, p0 = y[t == 1].mean(), y[t == 0].mean()
    se = np.sqrt(p1 * (1 - p1) / (t == 1).sum() + p0 * (1 - p0) / (t == 0).sum())
    return {"treated_rate": p1, "control_rate": p0, "ate": p1 - p0,
            "ci_low": p1 - p0 - 1.96 * se, "ci_high": p1 - p0 + 1.96 * se}


# ----------------------------------------------------------------------------- main
def main():
    plt = setup_matplotlib()
    raw = load()
    log(f"Hillstrom: {len(raw):,} customers | arms {raw.segment.value_counts().to_dict()}")

    # campaign-level ATEs for every arm and outcome (the classic A/B read-out)
    arm_ates = []
    for arm in ["Mens E-Mail", "Womens E-Mail"]:
        sub = raw[raw.segment.isin([arm, "No E-Mail"])]
        tt = (sub.segment == arm).astype(int).values
        for outcome in ["visit", "conversion"]:
            arm_ates.append({"arm": arm, "outcome": outcome,
                             **ate_with_ci(sub[outcome].values, tt)})
    arm_ates = pd.DataFrame(arm_ates)
    log(f"Average treatment effects:\n{arm_ates.round(4).to_string(index=False)}")

    X, t, y, _ = prepare(raw)
    strat = t * 2 + y
    Xtr, Xte, ttr, tte, ytr, yte = train_test_split(
        X, t, y, test_size=UPLIFT_TEST_SIZE, stratify=strat, random_state=RANDOM_STATE)
    log(f"Treatment='{UPLIFT_TREATMENT}', target='{UPLIFT_TARGET}' | "
        f"train {len(Xtr):,} test {len(Xte):,}")

    learners = {"S-learner": s_learner, "T-learner": t_learner,
                "Transformed outcome": transformed_outcome}
    scores = {name: fn(Xtr, ttr, ytr, Xte) for name, fn in learners.items()}
    rng = np.random.default_rng(RANDOM_STATE)
    scores["Random targeting"] = rng.random(len(yte))

    metric_rows, curve_rows, decile_rows = [], [], []
    n_treated_test = int(tte.sum())
    for name, s in scores.items():
        frac, q = qini_curve(yte, tte, s)
        row = {"model": name, "qini_coefficient": qini_coefficient(frac, q, n_treated_test)}
        for k in UPLIFT_TOP_K:
            row[f"uplift_at_{int(k * 100)}"] = uplift_at_k(yte, tte, s, k)
            idx = int(k * (len(q) - 1))
            row[f"share_of_total_gain_at_{int(k * 100)}"] = float(q[idx] / q[-1]) \
                if q[-1] > 0 else np.nan
        metric_rows.append(row)
        keep = np.linspace(0, len(frac) - 1, 201).astype(int)
        curve_rows += [{"model": name, "fraction_targeted": frac[i],
                        "incremental_outcomes": q[i]} for i in keep]
        if name != "Random targeting":
            decile_rows.append(uplift_by_decile(yte, tte, s).assign(model=name))

    metrics = pd.DataFrame(metric_rows).sort_values("qini_coefficient", ascending=False)
    log(f"Uplift metrics (test):\n{metrics.round(4).to_string(index=False)}")
    best = metrics[metrics.model != "Random targeting"].iloc[0]

    metrics.to_csv(OUTPUTS / "uplift_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUTPUTS / "uplift_qini_curves.csv", index=False)
    pd.concat(decile_rows).to_csv(OUTPUTS / "uplift_deciles.csv", index=False)
    arm_ates.to_csv(OUTPUTS / "uplift_arm_ates.csv", index=False)

    ate_test = ate_with_ci(yte, tte)
    k30 = "share_of_total_gain_at_30"
    save_json({
        "treatment": UPLIFT_TREATMENT, "target": UPLIFT_TARGET,
        "n_train": len(Xtr), "n_test": len(Xte),
        "ate_test": ate_test,
        "best_model": best.model,
        "best_qini_coefficient": best.qini_coefficient,
        "best_uplift_at_30": best.uplift_at_30,
        "best_share_of_gain_top30": best[k30],
        "headline": (f"Targeting the top 30% ranked by the {best.model} captures "
                     f"{best[k30]:.0%} of the incremental {UPLIFT_TARGET}s from mailing "
                     f"everyone, with uplift {best.uplift_at_30 * 100:.1f} pp vs an "
                     f"average of {ate_test['ate'] * 100:.1f} pp."),
    }, OUTPUTS / "uplift_summary.json")

    # ---- figures
    curves = pd.DataFrame(curve_rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, g) in enumerate(curves.groupby("model", sort=False)):
        style = "--" if name == "Random targeting" else "-"
        ax[0].plot(g.fraction_targeted, g.incremental_outcomes, style,
                   color="grey" if style == "--" else PALETTE[i], label=name)
    ax[0].set(title="Qini curve (test set)", xlabel="fraction of customers targeted",
              ylabel=f"incremental {UPLIFT_TARGET}s")
    ax[0].legend(fontsize=8)
    dec = pd.concat(decile_rows)
    d_best = dec[dec.model == best.model]
    ax[1].bar(d_best.decile, d_best.actual_uplift * 100, color=PALETTE[0])
    ax[1].axhline(ate_test["ate"] * 100, ls="--", color=PALETTE[3], label="average effect")
    ax[1].set(title=f"Actual uplift by predicted decile ({best.model})",
              xlabel="decile (1 = highest predicted uplift)", ylabel="uplift (pp)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "uplift_qini_deciles.png"); plt.close(fig)


if __name__ == "__main__":
    main()
