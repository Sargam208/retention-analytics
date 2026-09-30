"""
Step 2 - RFM segmentation with PCA + K-Means.

Pipeline: log1p (RFM is heavily right-skewed) -> StandardScaler -> PCA (keep 90% variance)
          -> K-Means for k in 2..8, pick k by silhouette within 3..7.
Clusters get business names from their RFM profile vs the customer-base median.

The customer table is ~5.9k rows, so scikit-learn is the right tool here; Spark did the
heavy lifting (1M+ rows -> 5.9k customers) in step 1.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline

from config import (
    FIGURES, FORCE_K, K_RANGE, K_SELECT_RANGE, MODELS, OUTPUTS, PCA_VARIANCE_TO_KEEP,
    RANDOM_STATE, SEGMENT_FEATURES, SNAPSHOT_CSV,
)
from utils import PALETTE, log, save_json, setup_matplotlib


def name_clusters(profile: pd.DataFrame, overall: pd.Series) -> dict:
    """Rule-based names from cluster medians vs population medians."""
    names = {}
    for cid, row in profile.iterrows():
        recent = row["recency_days"] <= overall["recency_days"]
        frequent = row["frequency"] >= overall["frequency"]
        valuable = row["monetary"] >= overall["monetary"]
        if recent and frequent and valuable:
            names[cid] = "Champions"
        elif recent and (frequent or valuable):
            names[cid] = "Loyal"
        elif recent:
            names[cid] = "New / Promising"
        elif frequent or valuable:
            names[cid] = "At Risk"
        else:
            names[cid] = "Hibernating"
    # disambiguate duplicates by value (e.g. two "Hibernating" clusters)
    s = pd.Series(names)
    for name, ids in s.groupby(s).groups.items():
        if len(ids) > 1:
            ranked = profile.loc[list(ids), "monetary"].sort_values(ascending=False).index
            for i, cid in enumerate(ranked):
                names[cid] = f"{name} {'ABCDEFG'[i]}"
    return names


def main():
    plt = setup_matplotlib()
    df = pd.read_csv(SNAPSHOT_CSV)
    X_raw = df[SEGMENT_FEATURES].clip(lower=0)
    log(f"Segmenting {len(df):,} customers on {SEGMENT_FEATURES}")

    prep = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
    X = prep.fit_transform(X_raw)

    pca_full = PCA(random_state=RANDOM_STATE).fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, PCA_VARIANCE_TO_KEEP) + 1)
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(X)
    Z = pca.transform(X)
    log(f"PCA keeps {n_comp} components ({cumvar[n_comp - 1]:.1%} variance)")

    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Z)
        rows.append({
            "k": k, "inertia": km.inertia_,
            "silhouette": silhouette_score(Z, km.labels_, sample_size=min(5000, len(Z)),
                                           random_state=RANDOM_STATE),
            "davies_bouldin": davies_bouldin_score(Z, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(Z, km.labels_),
        })
    k_table = pd.DataFrame(rows)
    cand = k_table[k_table.k.isin(list(K_SELECT_RANGE))]
    k_best = FORCE_K or int(cand.loc[cand.silhouette.idxmax(), "k"])
    log(f"k table:\n{k_table.round(3).to_string(index=False)}\n-> chosen k = {k_best}")

    km = KMeans(n_clusters=k_best, n_init=50, random_state=RANDOM_STATE).fit(Z)
    df["segment_id"] = km.labels_
    df["pc1"], df["pc2"] = Z[:, 0], Z[:, 1] if Z.shape[1] > 1 else 0.0

    overall = df[["recency_days", "frequency", "monetary"]].median()
    med = df.groupby("segment_id")[["recency_days", "frequency", "monetary"]].median()
    names = name_clusters(med, overall)
    df["segment"] = df.segment_id.map(names)

    total_rev = df.monetary.sum()
    profile = (
        df.groupby("segment")
        .agg(customers=("customer_id", "size"),
             median_recency_days=("recency_days", "median"),
             median_frequency=("frequency", "median"),
             median_monetary=("monetary", "median"),
             mean_avg_order_value=("avg_order_value", "mean"),
             median_tenure_days=("tenure_days", "median"),
             revenue=("monetary", "sum"))
        .assign(share_customers=lambda t: t.customers / t.customers.sum(),
                share_revenue=lambda t: t.revenue / total_rev)
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    log(f"Segment profile:\n{profile.round(2).to_string(index=False)}")

    # ---- save
    keep = ["customer_id", "segment_id", "segment", "pc1", "pc2"] + SEGMENT_FEATURES
    df[keep].to_csv(OUTPUTS / "customers_segmented.csv", index=False)
    profile.to_csv(OUTPUTS / "segment_profile.csv", index=False)
    k_table.to_csv(OUTPUTS / "segmentation_k_selection.csv", index=False)
    loadings = pd.DataFrame(pca.components_.T, index=SEGMENT_FEATURES,
                            columns=[f"PC{i + 1}" for i in range(n_comp)])
    loadings.to_csv(OUTPUTS / "pca_loadings.csv")
    joblib.dump({"prep": prep, "pca": pca, "kmeans": km, "names": names},
                MODELS / "segmentation.joblib")
    save_json({
        "n_customers": len(df), "features": SEGMENT_FEATURES,
        "pca_components": n_comp, "pca_variance_kept": float(cumvar[n_comp - 1]),
        "chosen_k": k_best,
        "silhouette": float(k_table.set_index("k").loc[k_best, "silhouette"]),
        "segments": profile.to_dict(orient="records"),
    }, OUTPUTS / "segmentation_metrics.json")

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(k_table.k, k_table.inertia, marker="o")
    ax[0].set(title="Elbow (inertia)", xlabel="k")
    ax[1].plot(k_table.k, k_table.silhouette, marker="o", color=PALETTE[1])
    ax[1].axvline(k_best, ls="--", color="grey")
    ax[1].set(title="Silhouette score", xlabel="k")
    fig.tight_layout(); fig.savefig(FIGURES / "seg_k_selection.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, g) in enumerate(df.groupby("segment")):
        ax.scatter(g.pc1, g.pc2, s=6, alpha=0.5, label=f"{name} ({len(g):,})",
                   color=PALETTE[i % len(PALETTE)])
    ax.set(title="Customer segments in PCA space", xlabel="PC1", ylabel="PC2")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "seg_pca_scatter.png"); plt.close(fig)


if __name__ == "__main__":
    main()
