"""Central configuration: paths and every analysis parameter in one place.

Change a number here, re-run the pipeline, and every script + the dashboard picks it up.
"""
from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
for _p in (DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# Raw inputs (see data/README.md for where to get them)
RETAIL_XLSX = DATA_RAW / "online_retail_II.xlsx"
RETAIL_CSV = DATA_RAW / "online_retail_II.csv"
COOKIE_CATS_CSV = DATA_RAW / "cookie_cats.csv"
HILLSTROM_CSV = DATA_RAW / "hillstrom.csv"

# Customer-level tables produced by the Spark ETL (small -> plain CSV)
SNAPSHOT_CSV = DATA_PROCESSED / "customers_snapshot.csv"   # features as of end of data
CHURN_TRAIN_CSV = DATA_PROCESSED / "churn_train.csv"       # features + label, earlier cutoff
CHURN_TEST_CSV = DATA_PROCESSED / "churn_test.csv"         # features + label, later cutoff
COHORT_CSV = DATA_PROCESSED / "cohort_counts.csv"
MONTHLY_KPI_CSV = DATA_PROCESSED / "monthly_kpis.csv"

RANDOM_STATE = 42

# ----------------------------------------------------------------------------- churn
# Churn has no label in the raw data, so we define it:
#   a customer who bought at least once in the ACTIVE_LOOKBACK_DAYS before the cutoff
#   is "churned" if they make NO purchase in the CHURN_HORIZON_DAYS after the cutoff.
# Features only use data strictly BEFORE the cutoff -> no leakage.
# Two cutoffs give a genuine out-of-time evaluation: train on the earlier snapshot,
# test on the later one (which the model has never seen).
CHURN_HORIZON_DAYS = 90
ACTIVE_LOOKBACK_DAYS = 365
TRAIN_CUTOFF = "2011-03-13"   # label window 2011-03-13 .. 2011-06-10
TEST_CUTOFF = "2011-09-11"    # label window 2011-09-11 .. 2011-12-09 (last day in data)

# ----------------------------------------------------------------------------- segmentation
SEGMENT_FEATURES = [
    "recency_days", "frequency", "monetary",
    "tenure_days", "avg_order_value", "n_products",
]
PCA_VARIANCE_TO_KEEP = 0.90
K_RANGE = range(2, 9)          # evaluated (elbow + silhouette)
K_SELECT_RANGE = range(3, 8)   # chosen from; k=2 is rarely useful for marketing
FORCE_K = None                 # set an int to override the silhouette choice

# ----------------------------------------------------------------------------- A/B test
AB_CONTROL = "gate_30"
AB_TREATMENT = "gate_40"
AB_METRICS = ["retention_1", "retention_7"]
AB_ALPHA = 0.05
AB_POWER = 0.80
N_BOOTSTRAP = 10_000

# ----------------------------------------------------------------------------- uplift
# "any" = Mens OR Womens e-mail vs No e-mail.  Or "Womens E-Mail" / "Mens E-Mail".
UPLIFT_TREATMENT = "any"
UPLIFT_TARGET = "visit"        # "visit" or "conversion"
UPLIFT_TEST_SIZE = 0.30
UPLIFT_TOP_K = [0.1, 0.2, 0.3, 0.5]
