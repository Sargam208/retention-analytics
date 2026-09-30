"""
Client Retention Analytics & Experimentation dashboard.

    streamlit run app/app.py          (from the repo root)

Reads only the files in outputs/ and data/processed/ written by the pipeline, so it
deploys to Streamlit Community Cloud without Spark or the raw data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed"

INK = "#16283A"
BLUE = "#1F6F8B"
AMBER = "#E3A33B"
GREEN = "#5E8C61"
RED = "#C8553D"
GREY = "#8A9BA8"
SEQ = [BLUE, AMBER, GREEN, RED, "#6C5B7B", GREY, "#2E4057"]
RISK_COLORS = {"High": RED, "Medium": AMBER, "Low": GREEN}

st.set_page_config(page_title="Retention Analytics", page_icon="📉", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  h1 {font-weight: 700; letter-spacing: -0.02em;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
  [data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #4A5B6B;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def csv(name, folder=OUT):
    p = folder / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def js(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def missing(step):
    st.info(f"No results yet. Run `python run_all.py --only {step}` and refresh.")


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=INK), plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(showgrid=False, linecolor="#C9D3DB")
    fig.update_yaxes(gridcolor="#EEF3F5")
    return fig


def chart(where, fig, h=380):
    """Full-width plotly chart; works on old and new Streamlit versions."""
    fig = style(fig, h)
    try:
        where.plotly_chart(fig, width="stretch")
    except TypeError:
        where.plotly_chart(fig, use_container_width=True)


def table(where, data):
    try:
        where.dataframe(data, hide_index=True, width="stretch")
    except TypeError:
        where.dataframe(data, hide_index=True, use_container_width=True)


st.title("Client retention analytics")
st.caption("Who is leaving, who is worth saving, and which interventions actually work.")

tab_over, tab_seg, tab_churn, tab_ab, tab_up = st.tabs(
    ["Overview", "Segments", "Churn risk", "A/B test", "Uplift targeting"])

# ============================================================================= overview
with tab_over:
    etl = js("etl_report.json")
    kpi = csv("monthly_kpis.csv", PROC)
    cohort = csv("cohort_counts.csv", PROC)
    if etl is None:
        missing(1)
    else:
        c = st.columns(4)
        c[0].metric("Transaction line items", f"{etl['raw_rows']:,}")
        c[1].metric("Customers", f"{etl['customers']:,}")
        c[2].metric("Revenue (£)", f"{etl['gross_revenue'] / 1e6:,.1f}M")
        c[3].metric("90-day churn (latest)", f"{etl['test_churn_rate']:.1%}")

        if kpi is not None:
            kpi["month"] = pd.to_datetime(kpi["month"])
            kpi = kpi[kpi.month < kpi.month.max()]          # last month is partial
            left, right = st.columns(2)
            f = px.bar(kpi, x="month", y="revenue", title="Monthly revenue (£)",
                       color_discrete_sequence=[BLUE])
            chart(left, f)
            f = px.line(kpi, x="month", y="active_customers", markers=True,
                        title="Active customers per month", color_discrete_sequence=[AMBER])
            chart(right, f)

        if cohort is not None:
            st.subheader("Cohort retention")
            st.caption("Share of each monthly acquisition cohort still buying N months later. "
                       "The Dec-2009 cohort includes pre-existing customers (data starts there).")
            size = cohort[cohort.period == 0].set_index("cohort").customers
            cohort["retention"] = cohort.customers / cohort.cohort.map(size)
            m = cohort.pivot(index="cohort", columns="period", values="retention")
            m = m.loc[:, [c for c in m.columns if c <= 12]]
            f = px.imshow(m, color_continuous_scale=["#FFFFFF", "#9CC5D3", BLUE, INK],
                          aspect="auto", text_auto=".0%",
                          labels=dict(x="months since first purchase", y="cohort",
                                      color="retained"))
            f.update_traces(textfont_size=9)
            chart(st, f, 560)

        with st.expander("Data cleaning audit trail"):
            table(st, pd.DataFrame(etl["cleaning_steps"]))

# ============================================================================= segments
with tab_seg:
    seg = csv("customers_segmented.csv")
    prof = csv("segment_profile.csv")
    ksel = csv("segmentation_k_selection.csv")
    meta = js("segmentation_metrics.json")
    if seg is None:
        missing(2)
    else:
        st.markdown(f"**{meta['chosen_k']} segments** from K-Means on "
                    f"{meta['pca_components']} principal components "
                    f"({meta['pca_variance_kept']:.0%} of variance kept, "
                    f"silhouette {meta['silhouette']:.2f}).")
        left, right = st.columns([3, 2])
        sample = seg.sample(min(4000, len(seg)), random_state=0)
        f = px.scatter(sample, x="pc1", y="pc2", color="segment", opacity=0.6,
                       color_discrete_sequence=SEQ, title="Customers in PCA space",
                       hover_data=["customer_id", "recency_days", "frequency", "monetary"])
        f.update_traces(marker_size=5)
        chart(left, f, 440)

        share = prof.melt(id_vars="segment", value_vars=["share_customers", "share_revenue"],
                          var_name="share", value_name="value")
        share["share"] = share.share.map({"share_customers": "% of customers",
                                          "share_revenue": "% of revenue"})
        f = px.bar(share, y="segment", x="value", color="share", barmode="group",
                   orientation="h", color_discrete_sequence=[GREY, BLUE],
                   title="Who drives the revenue")
        f.update_xaxes(tickformat=".0%")
        chart(right, f, 440)

        show = prof[["segment", "customers", "median_recency_days", "median_frequency",
                     "median_monetary", "share_revenue"]].copy()
        show.columns = ["Segment", "Customers", "Median days since last order",
                        "Median orders", "Median spend (£)", "Revenue share"]
        table(st, show.style.format({"Median spend (£)": "{:,.0f}",
                                        "Revenue share": "{:.1%}",
                                        "Median days since last order": "{:.0f}",
                                        "Median orders": "{:.0f}"}))

        if ksel is not None:
            with st.expander("How k was chosen"):
                f = go.Figure()
                f.add_scatter(x=ksel.k, y=ksel.silhouette, mode="lines+markers",
                              name="silhouette", line_color=BLUE)
                f.add_vline(x=meta["chosen_k"], line_dash="dash", line_color=GREY)
                chart(st, f, 300)

# ============================================================================= churn
with tab_churn:
    met = csv("churn_metrics.csv")
    summ = js("churn_summary.json")
    scores = csv("churn_scores_current.csv")
    if met is None:
        missing(3)
    else:
        best = met.set_index("model").loc[summ["best_model"]]
        st.markdown(f"Trained on the {js('etl_report.json')['train_cutoff']} snapshot, tested "
                    f"on a later snapshot it never saw. Best model: **{summ['best_model']}**.")
        c = st.columns(4)
        c[0].metric("ROC-AUC", f"{best.roc_auc:.3f}",
                    f"{best.roc_auc - summ['recency_only_baseline']['roc_auc']:+.3f} vs recency rule")
        c[1].metric("PR-AUC", f"{best.pr_auc:.3f}")
        c[2].metric("Churners caught in top 20%", f"{best.capture_top20:.0%}")
        c[3].metric("Lift in top 20%", f"{best.lift_top20:.2f}x")

        curves = csv("churn_curves.csv")
        left, right = st.columns(2)
        if curves is not None:
            r = curves[curves.curve == "roc"]
            f = px.line(r, x="x", y="y", color="model", color_discrete_sequence=SEQ,
                        title="ROC curve", labels={"x": "false positive rate",
                                                   "y": "true positive rate"})
            f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=GREY))
            chart(left, f)
        imp = csv("churn_feature_importance.csv")
        if imp is not None:
            top = imp.head(10)[::-1]
            f = px.bar(top, x="importance", y="feature", orientation="h",
                       color_discrete_sequence=[BLUE],
                       title="What predicts churn (permutation importance)")
            chart(right, f)

        table(st, met[["model", "cv_pr_auc", "roc_auc", "pr_auc", "brier", "f1",
                          "capture_top20", "lift_top20"]].round(3))

        if scores is not None:
            st.subheader("Customers to contact now")
            c1, c2 = st.columns(2)
            bands = c1.multiselect("Risk band", ["High", "Medium", "Low"], default=["High"])
            segs = sorted(scores.segment.dropna().unique()) if "segment" in scores else []
            chosen = c2.multiselect("Segment", segs, default=segs)
            view = scores[scores.risk_band.isin(bands)]
            if segs:
                view = view[view.segment.isin(chosen)]
            st.caption(f"{len(view):,} customers, £{view.monetary.sum():,.0f} of historical "
                       "spend. Sorted by churn probability.")
            table(st, view.head(500).style.format({"churn_probability": "{:.0%}",
                                                      "monetary": "£{:,.0f}",
                                                      "monetary_90d": "£{:,.0f}",
                                                      "avg_order_value": "£{:,.0f}"}))
            st.download_button("Download list as CSV", view.to_csv(index=False),
                               "churn_risk_customers.csv", "text/csv")

# ============================================================================= A/B
with tab_ab:
    ab = js("ab_summary.json")
    rates = csv("ab_rates.csv")
    boots = csv("ab_bootstrap_samples.csv")
    if ab is None:
        missing(4)
    else:
        srm = ab["srm"]
        st.markdown(f"**{ab['n_users']:,} players**, randomly assigned to the first gate at "
                    "level 30 (control) or level 40. Sample ratio check: "
                    f"p = {srm['p_value']:.2f} "
                    f"({'mismatch - investigate' if srm['srm_detected'] else 'split is healthy'}).")
        st.success(ab["decision"])

        res = pd.DataFrame(ab["results"])
        cols = st.columns(len(res))
        for col, r in zip(cols, res.itertuples()):
            col.metric(r.metric.replace("_", " ").replace("retention", "Day") + " retention",
                       f"{r.treatment_rate:.2%}", f"{r.abs_diff * 100:+.2f} pp vs control",
                       delta_color="normal")
            col.caption(f"95% bootstrap CI [{r.ci_boot_low * 100:+.2f}, "
                        f"{r.ci_boot_high * 100:+.2f}] pp · p = {r.p_value_z:.4f} "
                        f"(Holm {r.p_value_holm:.4f}) · MDE {r.mde_abs_at_80_power * 100:.2f} pp")

        left, right = st.columns(2)
        if rates is not None:
            f = px.bar(rates, x="metric", y="rate", color="version", barmode="group",
                       color_discrete_sequence=[GREY, BLUE], title="Retention by group",
                       text_auto=".1%")
            f.update_yaxes(tickformat=".0%")
            chart(left, f)
        if boots is not None:
            metric = right.selectbox("Bootstrap distribution", list(boots.columns))
            f = px.histogram(boots[metric] * 100, nbins=60, color_discrete_sequence=[BLUE],
                             title=f"Bootstrap: difference in {metric} (pp)")
            f.add_vline(x=0, line_color=INK)
            f.update_layout(showlegend=False)
            chart(right, f)

        g = ab["engagement_guardrail"]
        with st.expander("Engagement guardrail and data quality"):
            st.write(f"Median game rounds: {g['median_control']:.0f} (control) vs "
                     f"{g['median_treatment']:.0f} (treatment), Mann-Whitney p = "
                     f"{g['mannwhitney_p']:.3f}. Rounds are heavy-tailed, so medians and a "
                     "rank test are used instead of a t-test.")
            st.write(f"{ab['data_quality']['share_never_played']:.1%} of players never "
                     f"played a round; {ab['data_quality']['extreme_outliers']} extreme outlier(s).")

        with st.expander("Plan your next test"):
            st.caption("Sample size per group for a two-proportion test (two-sided).")
            c1, c2, c3 = st.columns(3)
            base = c1.number_input("Baseline rate", 0.01, 0.99, float(res.iloc[-1].control_rate),
                                   0.01, format="%.3f")
            mde = c2.number_input("Minimum detectable effect (pp)", 0.1, 20.0, 1.0, 0.1)
            power = c3.slider("Power", 0.5, 0.99, 0.8, 0.01)
            from scipy.stats import norm
            p2 = base + mde / 100
            h = abs(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(base)))
            n = ((norm.ppf(1 - ab["alpha"] / 2) + norm.ppf(power)) / h) ** 2
            st.metric("Users needed per group", f"{int(np.ceil(n)):,}")

# ============================================================================= uplift
with tab_up:
    up = js("uplift_summary.json")
    um = csv("uplift_metrics.csv")
    qc = csv("uplift_qini_curves.csv")
    dec = csv("uplift_deciles.csv")
    arms = csv("uplift_arm_ates.csv")
    if up is None:
        missing(5)
    else:
        st.markdown("A churn score says who might leave. Uplift says **whose behaviour the "
                    "campaign actually changes**, so budget is not spent on customers who "
                    "would have come back anyway.")
        st.success(up["headline"])

        left, right = st.columns(2)
        if qc is not None:
            f = px.line(qc, x="fraction_targeted", y="incremental_outcomes", color="model",
                        color_discrete_sequence=[BLUE, AMBER, GREEN, GREY],
                        title="Qini curve: incremental visits by share targeted",
                        labels={"fraction_targeted": "share of customers e-mailed",
                                "incremental_outcomes": "incremental visits"})
            f.update_xaxes(tickformat=".0%")
            chart(left, f)
        if dec is not None:
            model = right.selectbox("Model", dec.model.unique().tolist(),
                                    index=dec.model.unique().tolist().index(up["best_model"]))
            d = dec[dec.model == model]
            f = px.bar(d, x="decile", y="actual_uplift", color_discrete_sequence=[BLUE],
                       title="Observed uplift by predicted-uplift decile")
            f.add_hline(y=up["ate_test"]["ate"], line_dash="dash", line_color=RED,
                        annotation_text="average effect")
            f.update_yaxes(tickformat=".1%")
            chart(right, f)

        table(st, um.round(4))
        if arms is not None:
            with st.expander("Campaign-level results (classic A/B read-out)"):
                table(st, arms.round(4))
