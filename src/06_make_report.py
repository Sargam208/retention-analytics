"""
Step 6 - Collect every number into outputs/REPORT.md (paste into README / use for interviews).
"""
import pandas as pd

from config import OUTPUTS
from utils import load_json, log


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def pp(x, d=2):
    return f"{x * 100:+.{d}f} pp"


def section_etl(lines):
    p = OUTPUTS / "etl_report.json"
    if not p.exists():
        return
    e = load_json(p)
    lines += [
        "## 1. Data pipeline (PySpark)",
        f"- Raw line items: **{e['raw_rows']:,}** -> clean: **{e['clean_rows']:,}** "
        f"({e['data_start']} to {e['data_end']})",
        f"- {e['customers']:,} customers, {e['invoices']:,} invoices, "
        f"{e['products']:,} products, {e['countries']} countries",
        f"- Churn = no purchase in the {e['churn_horizon_days']} days after the cutoff "
        f"(customers active in the previous {e['active_lookback_days']} days)",
        f"- Train @ {e['train_cutoff']}: {e['train_customers']:,} customers, churn "
        f"{pct(e['train_churn_rate'])} | Test @ {e['test_cutoff']}: "
        f"{e['test_customers']:,} customers, churn {pct(e['test_churn_rate'])}",
        "", "| Cleaning step | Rows after | Removed |", "|---|---:|---:|",
    ]
    lines += [f"| {s['step']} | {s['rows_after']:,} | {s['rows_removed']:,} |"
              for s in e["cleaning_steps"]]
    lines.append("")


def section_seg(lines):
    p = OUTPUTS / "segmentation_metrics.json"
    if not p.exists():
        return
    s = load_json(p)
    lines += [
        "## 2. Segmentation (PCA + K-Means)",
        f"- {s['pca_components']} principal components keep {pct(s['pca_variance_kept'])} "
        f"of variance; k = {s['chosen_k']} (silhouette {s['silhouette']:.3f})",
        "", "| Segment | Customers | % customers | % revenue | Median recency (d) | "
        "Median orders | Median spend |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {g['segment']} | {g['customers']:,} | {pct(g['share_customers'])} | "
              f"{pct(g['share_revenue'])} | {g['median_recency_days']:.0f} | "
              f"{g['median_frequency']:.0f} | {g['median_monetary']:,.0f} |"
              for g in s["segments"]]
    lines.append("")


def section_churn(lines):
    p = OUTPUTS / "churn_metrics.csv"
    if not p.exists():
        return
    m = pd.read_csv(p)
    s = load_json(OUTPUTS / "churn_summary.json")
    b = s["recency_only_baseline"]
    lines += [
        "## 3. Churn model (out-of-time test)",
        "| Model | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Brier | F1 | "
        "Capture @ top 20% | Lift @ top 20% |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {r.model} | {r.cv_pr_auc:.3f} | {r.roc_auc:.3f} | {r.pr_auc:.3f} | "
              f"{r.brier:.3f} | {r.f1:.3f} | {pct(r.capture_top20)} | {r.lift_top20:.2f}x |"
              for r in m.itertuples()]
    lines += [
        f"| Recency-only rule (baseline) | - | {b['roc_auc']:.3f} | {b['pr_auc']:.3f} | "
        "- | - | - | - |",
        "", f"- Best: **{s['best_model']}**. Top drivers: {', '.join(s['top_features'][:5])}",
        f"- Scored {s['n_scored_current']:,} currently active customers "
        "(outputs/churn_scores_current.csv)", "",
    ]


def section_ab(lines):
    p = OUTPUTS / "ab_summary.json"
    if not p.exists():
        return
    a = load_json(p)
    srm = a["srm"]
    lines += [
        "## 4. A/B test (Cookie Cats)",
        f"- {a['n_users']:,} users; SRM chi-square p = {srm['p_value']:.3f} "
        f"({'mismatch!' if srm['srm_detected'] else 'randomisation OK'})",
        "", "| Metric | Control | Treatment | Diff | 95% bootstrap CI | p (z) | p (Holm) | "
        "MDE @80% power |", "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    lines += [f"| {r['metric']} | {pct(r['control_rate'], 2)} | {pct(r['treatment_rate'], 2)} "
              f"| {pp(r['abs_diff'])} | [{pp(r['ci_boot_low'])}, {pp(r['ci_boot_high'])}] | "
              f"{r['p_value_z']:.4f} | {r['p_value_holm']:.4f} | "
              f"{r['mde_abs_at_80_power'] * 100:.2f} pp |" for r in a["results"]]
    g = a["engagement_guardrail"]
    lines += ["", f"- Engagement guardrail: median rounds {g['median_control']:.0f} vs "
              f"{g['median_treatment']:.0f} (Mann-Whitney p = {g['mannwhitney_p']:.3f})",
              f"- **Decision:** {a['decision']}", ""]


def section_uplift(lines):
    p = OUTPUTS / "uplift_metrics.csv"
    if not p.exists():
        return
    m = pd.read_csv(p)
    s = load_json(OUTPUTS / "uplift_summary.json")
    ate = s["ate_test"]
    lines += [
        "## 5. Uplift modelling (Hillstrom)",
        f"- Average effect on {s['target']} (test): {pp(ate['ate'])} "
        f"[{pp(ate['ci_low'])}, {pp(ate['ci_high'])}]",
        "", "| Model | Qini coef. | Uplift @10% | Uplift @30% | Share of total gain @30% |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [f"| {r.model} | {r.qini_coefficient:.4f} | {pp(r.uplift_at_10)} | "
              f"{pp(r.uplift_at_30)} | {pct(r.share_of_total_gain_at_30, 0)} |"
              for r in m.itertuples()]
    lines += ["", f"- **{s['headline']}**", ""]


def main():
    lines = ["# Results report", "_Auto-generated by `src/06_make_report.py`._", ""]
    for f in (section_etl, section_seg, section_churn, section_ab, section_uplift):
        f(lines)
    (OUTPUTS / "REPORT.md").write_text("\n".join(lines))
    log(f"saved {OUTPUTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
