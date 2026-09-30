"""
Step 3 - Churn prediction: Logistic Regression vs Random Forest vs XGBoost.

Design:
  * Train on the TRAIN_CUTOFF snapshot, evaluate on the later TEST_CUTOFF snapshot
    (out-of-time). A random split would let the model see the same period it's tested on.
  * Hyper-parameters + decision threshold are chosen with 5-fold CV on TRAIN only.
  * Metrics: ROC-AUC, PR-AUC (churn is not rare here but PR-AUC is what matters for a
    retention team with a limited budget), Brier score, and lift / capture in the top 20%.
  * The best model is refit on train+test and scores the CURRENT customer base, merged
    with segments -> the actionable list the dashboard shows.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from config import (
    ACTIVE_LOOKBACK_DAYS, CHURN_TEST_CSV, CHURN_TRAIN_CSV, FIGURES, MODELS, OUTPUTS,
    RANDOM_STATE, SNAPSHOT_CSV,
)
from utils import PALETTE, log, save_json, setup_matplotlib

FEATURES = [
    "recency_days", "frequency", "monetary", "tenure_days", "avg_order_value",
    "avg_items_per_order", "n_products", "n_purchase_days", "n_active_months",
    "frequency_90d", "monetary_90d", "monetary_prev_90d", "spend_trend",
    "avg_days_between_purchases", "is_one_time_buyer", "n_return_invoices",
    "return_rate", "is_uk",
]
TARGET = "target_churned"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_models(pos_weight: float) -> dict:
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(signed_log1p)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=0.5)),
    ])
    rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    xgb_base = XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        scale_pos_weight=pos_weight, eval_metric="logloss", tree_method="hist",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    xgb = RandomizedSearchCV(
        xgb_base,
        param_distributions={
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [200, 400, 600, 800],
            "min_child_weight": [1, 5, 10, 20],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
        n_iter=25, scoring="average_precision", cv=CV, random_state=RANDOM_STATE,
        n_jobs=-1, refit=True,
    )
    return {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])])


def lift_table(y, p, n_bins=10):
    d = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = np.arange(len(d)) * n_bins // len(d) + 1
    t = d.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"),
                                avg_score=("p", "mean"))
    t["churn_rate"] = t.churners / t.customers
    t["lift"] = t.churn_rate / d.y.mean()
    t["cum_capture"] = t.churners.cumsum() / d.y.sum()
    return t.reset_index()


def evaluate(name, y, p, thr):
    yhat = (p >= thr).astype(int)
    top = np.argsort(-p)[: int(0.2 * len(p))]
    return {
        "model": name,
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "threshold": thr,
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "precision_top20": float(y[top].mean()),
        "capture_top20": float(y[top].sum() / y.sum()),
        "lift_top20": float(y[top].mean() / y.mean()),
    }


def unwrap(model):
    return model.best_estimator_ if hasattr(model, "best_estimator_") else model


def main():
    plt = setup_matplotlib()
    train = pd.read_csv(CHURN_TRAIN_CSV)
    test = pd.read_csv(CHURN_TEST_CSV)
    X_tr, y_tr = train[FEATURES], train[TARGET].values
    X_te, y_te = test[FEATURES], test[TARGET].values
    log(f"train {len(train):,} (churn {y_tr.mean():.1%}) | "
        f"test {len(test):,} (churn {y_te.mean():.1%})")

    pos_weight = (1 - y_tr.mean()) / y_tr.mean()
    models = build_models(pos_weight)

    results, cv_rows, test_probs, fitted = [], [], {}, {}
    for name, model in models.items():
        log(f"Fitting {name}")
        model.fit(X_tr, y_tr)
        est = unwrap(model)
        fitted[name] = est
        # out-of-fold probabilities on TRAIN -> CV score + threshold (no test peeking)
        oof = cross_val_predict(clone(est), X_tr, y_tr, cv=CV, method="predict_proba")[:, 1]
        thr = best_f1_threshold(y_tr, oof)
        cv_rows.append({"model": name, "cv_roc_auc": roc_auc_score(y_tr, oof),
                        "cv_pr_auc": average_precision_score(y_tr, oof)})
        p = est.predict_proba(X_te)[:, 1]
        test_probs[name] = p
        results.append(evaluate(name, y_te, p, thr))
        if hasattr(model, "best_params_"):
            log(f"  best XGB params: {model.best_params_}")

    metrics = pd.DataFrame(results).merge(pd.DataFrame(cv_rows), on="model")
    metrics["baseline_churn_rate"] = y_te.mean()
    log(f"Out-of-time test metrics:\n{metrics.round(3).to_string(index=False)}")
    metrics.to_csv(OUTPUTS / "churn_metrics.csv", index=False)

    best_name = metrics.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    log(f"Best model by test PR-AUC: {best_name}")

    # ---- simple recency-only baseline: "does ML beat a rule?"
    base_auc = roc_auc_score(y_te, test.recency_days)
    base_ap = average_precision_score(y_te, test.recency_days)

    # ---- explanations
    imp = permutation_importance(best, X_te, y_te, scoring="average_precision",
                                 n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("importance", ascending=False))
    importance.to_csv(OUTPUTS / "churn_feature_importance.csv", index=False)
    try:
        import shap
        xgb_est = fitted["XGBoost"]
        sv = shap.TreeExplainer(xgb_est).shap_values(X_te)
        shap.summary_plot(sv, X_te, show=False, max_display=12)
        plt.gcf().tight_layout(); plt.savefig(FIGURES / "churn_shap_summary.png"); plt.close()
        log("saved SHAP summary")
    except Exception as e:  # shap is optional
        log(f"(SHAP skipped: {type(e).__name__})")

    lift = lift_table(y_te, test_probs[best_name])
    lift.to_csv(OUTPUTS / "churn_lift_table.csv", index=False)

    # ---- curves for the dashboard
    curves = []
    for name, p in test_probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        prec, rec, _ = precision_recall_curve(y_te, p)
        idx_r = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
        idx_p = np.linspace(0, len(prec) - 1, min(200, len(prec))).astype(int)
        curves += [{"model": name, "curve": "roc", "x": fpr[i], "y": tpr[i]} for i in idx_r]
        curves += [{"model": name, "curve": "pr", "x": rec[i], "y": prec[i]} for i in idx_p]
    pd.DataFrame(curves).to_csv(OUTPUTS / "churn_curves.csv", index=False)

    # ---- refit on all labelled data, score today's active customers
    full = pd.concat([train, test], ignore_index=True)
    final = clone(best).fit(full[FEATURES], full[TARGET])
    joblib.dump({"model": final, "features": FEATURES, "name": best_name},
                MODELS / "churn_model.joblib")
    snap = pd.read_csv(SNAPSHOT_CSV)
    snap = snap[snap.recency_days <= ACTIVE_LOOKBACK_DAYS].copy()
    snap["churn_probability"] = final.predict_proba(snap[FEATURES])[:, 1]
    snap["risk_band"] = pd.cut(snap.churn_probability, [0, 0.3, 0.6, 1.0],
                               labels=["Low", "Medium", "High"], include_lowest=True)
    seg_path = OUTPUTS / "customers_segmented.csv"
    if seg_path.exists():
        snap = snap.merge(pd.read_csv(seg_path)[["customer_id", "segment"]],
                          on="customer_id", how="left")
    cols = ["customer_id", "segment", "churn_probability", "risk_band", "recency_days",
            "frequency", "monetary", "monetary_90d", "avg_order_value"]
    snap[[c for c in cols if c in snap]].sort_values("churn_probability", ascending=False) \
        .to_csv(OUTPUTS / "churn_scores_current.csv", index=False)
    log(f"Scored {len(snap):,} active customers "
        f"({(snap.risk_band == 'High').mean():.1%} high risk)")

    top = metrics.set_index("model").loc[best_name]
    save_json({
        "best_model": best_name,
        "test": top.to_dict(),
        "recency_only_baseline": {"roc_auc": base_auc, "pr_auc": base_ap},
        "top_features": importance.head(8).feature.tolist(),
        "n_train": len(train), "n_test": len(test),
        "n_scored_current": len(snap),
    }, OUTPUTS / "churn_summary.json")

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, (name, p) in enumerate(test_probs.items()):
        fpr, tpr, _ = roc_curve(y_te, p)
        prec, rec, _ = precision_recall_curve(y_te, p)
        ax[0].plot(fpr, tpr, color=PALETTE[i], label=f"{name} ({roc_auc_score(y_te, p):.3f})")
        ax[1].plot(rec, prec, color=PALETTE[i],
                   label=f"{name} ({average_precision_score(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", color="grey")
    ax[1].axhline(y_te.mean(), ls="--", color="grey")
    ax[0].set(title="ROC (out-of-time test)", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "churn_roc_pr.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    top_imp = importance.head(12)[::-1]
    ax.barh(top_imp.feature, top_imp.importance, xerr=top_imp["std"], color=PALETTE[0])
    ax.set(title=f"Permutation importance ({best_name})", xlabel="drop in PR-AUC")
    fig.tight_layout(); fig.savefig(FIGURES / "churn_feature_importance.png"); plt.close(fig)


if __name__ == "__main__":
    main()
