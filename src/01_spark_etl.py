"""
Step 1 - PySpark ETL on UCI Online Retail II (~1.07M line items).

  raw line items --clean--> transactions --aggregate--> customer feature tables

Outputs (all small, customer-level, so saved as CSV via pandas):
  customers_snapshot.csv   features as of the last day of data   -> segmentation + live scoring
  churn_train.csv          features @TRAIN_CUTOFF + churn label  -> model training
  churn_test.csv           features @TEST_CUTOFF  + churn label  -> out-of-time test
  cohort_counts.csv        monthly acquisition cohorts x months since first purchase
  monthly_kpis.csv         revenue / orders / active customers per month
  outputs/etl_report.json  row counts at every cleaning step (data-quality audit trail)

Why only CSV (no Spark parquet writes)? Spark writing files on Windows needs winutils.exe;
the customer tables are ~5k rows, so collecting them to pandas is cheap and portable.
"""
import datetime as dt

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from config import (
    ACTIVE_LOOKBACK_DAYS, CHURN_HORIZON_DAYS, CHURN_TEST_CSV, CHURN_TRAIN_CSV,
    COHORT_CSV, MONTHLY_KPI_CSV, OUTPUTS, RETAIL_CSV, RETAIL_XLSX, SNAPSHOT_CSV,
    TEST_CUTOFF, TRAIN_CUTOFF,
)
from utils import get_spark, log, save_json

RENAME = {
    "Invoice": "invoice", "InvoiceNo": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price", "UnitPrice": "price",
    "Customer ID": "customer_id", "CustomerID": "customer_id",
    "Country": "country",
}
REQUIRED = {"invoice", "stock_code", "quantity", "invoice_date", "price", "customer_id", "country"}


# ----------------------------------------------------------------------------- load
def ensure_csv():
    """UCI ships an .xlsx with two sheets (2009-10, 2010-11). Spark can't read xlsx,
    so convert once with pandas. Kaggle's CSV copy can be dropped in directly."""
    if RETAIL_CSV.exists():
        return RETAIL_CSV
    if RETAIL_XLSX.exists():
        log("Converting online_retail_II.xlsx -> csv (one-time, takes a few minutes)...")
        sheets = pd.read_excel(RETAIL_XLSX, sheet_name=None,
                               dtype={"Invoice": str, "StockCode": str})
        pd.concat(sheets.values(), ignore_index=True).to_csv(RETAIL_CSV, index=False)
        return RETAIL_CSV
    raise FileNotFoundError(
        f"Put online_retail_II.xlsx or online_retail_II.csv in {RETAIL_CSV.parent} "
        "(see data/README.md)."
    )


def load_raw(spark, path) -> DataFrame:
    df = spark.read.csv(str(path), header=True, inferSchema=False, escape='"')
    for old, new in RENAME.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {df.columns}")

    ts = F.coalesce(
        F.to_timestamp("invoice_date", "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp("invoice_date", "yyyy-MM-dd HH:mm"),
        F.to_timestamp("invoice_date", "M/d/yyyy H:mm"),
    )
    return (
        df.withColumn("invoice", F.trim(F.col("invoice")))
        .withColumn("stock_code", F.upper(F.trim(F.col("stock_code"))))
        .withColumn("quantity", F.col("quantity").cast("double").cast("long"))
        .withColumn("price", F.col("price").cast("double"))
        # xlsx -> pandas turns IDs into "13085.0"; double -> long handles both forms
        .withColumn("customer_id", F.col("customer_id").cast("double").cast("long"))
        .withColumn("invoice_ts", ts)
        .withColumn("country", F.trim(F.col("country")))
        .select("invoice", "stock_code", "quantity", "invoice_ts", "price",
                "customer_id", "country")
    )


# ----------------------------------------------------------------------------- clean
def clean(df: DataFrame):
    """Every filter is logged with before/after counts -> etl_report.json."""
    steps = []

    def step(name, new_df, prev_n):
        n = new_df.count()
        steps.append({"step": name, "rows_after": n, "rows_removed": prev_n - n})
        log(f"  {name:<55} {n:>10,}  (-{prev_n - n:,})")
        return new_df, n

    n = df.count()
    steps.append({"step": "raw line items", "rows_after": n, "rows_removed": 0})
    log(f"  {'raw line items':<55} {n:>10,}")

    # The two UCI sheets overlap in early Dec-2010, so exact duplicates exist.
    df, n = step("drop exact duplicate rows", df.dropDuplicates(), n)
    df, n = step("drop rows without customer_id (guest checkouts)",
                 df.filter(F.col("customer_id").isNotNull()), n)
    df, n = step("drop unparseable dates", df.filter(F.col("invoice_ts").isNotNull()), n)
    df, n = step("drop price <= 0 (free samples, bad-debt adjustments)",
                 df.filter(F.col("price") > 0), n)
    # Real products have 5-digit codes (e.g. 85123A). POST, DOT, M, BANK CHARGES,
    # AMAZONFEE, C2, ... are postage / fees / manual adjustments, not purchases.
    df, n = step("keep product stock codes only",
                 df.filter(F.col("stock_code").rlike("^[0-9]{5}")), n)

    df = df.withColumn("is_return", F.col("invoice").startswith("C"))
    consistent = ((~F.col("is_return")) & (F.col("quantity") > 0)) | \
                 (F.col("is_return") & (F.col("quantity") < 0))
    df, n = step("drop sign-inconsistent quantity rows", df.filter(consistent), n)

    df = (df.withColumn("revenue", F.col("quantity") * F.col("price"))   # < 0 for returns
            .withColumn("invoice_day", F.to_date("invoice_ts")))
    return df, steps


# ----------------------------------------------------------------------------- features
def build_customer_features(tx: DataFrame, cutoff: dt.date, horizon_days=None,
                            lookback_days=None) -> DataFrame:
    """Customer features using ONLY transactions strictly before `cutoff`.

    If horizon_days is given, also attaches labels from [cutoff, cutoff + horizon):
      target_churned        1 if no purchase in the window
      target_future_revenue revenue in the window (for business evaluation only)
    Columns prefixed target_ must never be used as model inputs.
    """
    c = F.lit(cutoff).cast("date")
    hist = tx.filter(F.col("invoice_day") < c)
    purch = hist.filter(~F.col("is_return"))
    rets = hist.filter(F.col("is_return"))

    orders = purch.groupBy("customer_id", "invoice").agg(
        F.min("invoice_day").alias("day"),
        F.sum("revenue").alias("order_value"),
        F.sum("quantity").alias("items"),
    )
    d = F.col("day")
    last90 = d >= F.date_sub(c, 90)
    prev90 = (d >= F.date_sub(c, 180)) & (d < F.date_sub(c, 90))

    cust = orders.groupBy("customer_id").agg(
        F.max("day").alias("last_day"),
        F.min("day").alias("first_day"),
        F.countDistinct("invoice").alias("frequency"),
        F.sum("order_value").alias("monetary"),
        F.sum("items").alias("total_items"),
        F.countDistinct("day").alias("n_purchase_days"),
        F.countDistinct(F.date_format("day", "yyyy-MM")).alias("n_active_months"),
        F.sum(F.when(last90, 1).otherwise(0)).alias("frequency_90d"),
        F.sum(F.when(last90, F.col("order_value")).otherwise(0.0)).alias("monetary_90d"),
        F.sum(F.when(prev90, F.col("order_value")).otherwise(0.0)).alias("monetary_prev_90d"),
    )

    products = purch.groupBy("customer_id").agg(
        F.countDistinct("stock_code").alias("n_products"),
        F.max(F.when(F.col("country") == "United Kingdom", 1).otherwise(0)).alias("is_uk"),
    )
    returns = rets.groupBy("customer_id").agg(
        F.countDistinct("invoice").alias("n_return_invoices"),
        F.sum(F.abs(F.col("revenue"))).alias("returned_value"),
    )

    feat = (
        cust.join(products, "customer_id", "left")
        .join(returns, "customer_id", "left")
        .fillna(0, subset=["n_return_invoices", "returned_value"])
        .withColumn("recency_days", F.datediff(c, F.col("last_day")))
        .withColumn("tenure_days", F.datediff(c, F.col("first_day")))
        .withColumn("avg_order_value", F.col("monetary") / F.col("frequency"))
        .withColumn("avg_items_per_order", F.col("total_items") / F.col("frequency"))
        .withColumn(
            "avg_days_between_purchases",
            F.when(F.col("n_purchase_days") > 1,
                   F.datediff(F.col("last_day"), F.col("first_day"))
                   / (F.col("n_purchase_days") - 1)),          # NULL for one-time buyers
        )
        .withColumn("is_one_time_buyer", (F.col("n_purchase_days") == 1).cast("int"))
        .withColumn("return_rate",
                    F.col("returned_value") / (F.col("monetary") + F.col("returned_value")))
        # log-ratio of spend in the last 90d vs the 90d before: <0 means slowing down
        .withColumn("spend_trend",
                    F.log1p(F.col("monetary_90d")) - F.log1p(F.col("monetary_prev_90d")))
        .drop("last_day", "first_day", "total_items")
    )

    if lookback_days is not None:
        feat = feat.filter(F.col("recency_days") <= lookback_days)

    if horizon_days is not None:
        future = (
            tx.filter((~F.col("is_return"))
                      & (F.col("invoice_day") >= c)
                      & (F.col("invoice_day") < F.date_add(c, horizon_days)))
            .groupBy("customer_id")
            .agg(F.sum("revenue").alias("target_future_revenue"))
        )
        feat = (
            feat.join(future, "customer_id", "left")
            .fillna(0.0, subset=["target_future_revenue"])
            .withColumn("target_churned",
                        (F.col("target_future_revenue") <= 0).cast("int"))
        )
    return feat


# ----------------------------------------------------------------------------- cohorts & KPIs
def cohort_counts(tx: DataFrame) -> DataFrame:
    purch = tx.filter(~F.col("is_return"))
    first = purch.groupBy("customer_id").agg(
        F.trunc(F.min("invoice_day"), "month").alias("cohort"))
    active = purch.select("customer_id",
                          F.trunc("invoice_day", "month").alias("month")).distinct()
    return (
        active.join(first, "customer_id")
        .withColumn("period", F.round(F.months_between("month", "cohort")).cast("int"))
        .groupBy("cohort", "period")
        .agg(F.countDistinct("customer_id").alias("customers"))
        .orderBy("cohort", "period")
    )


def monthly_kpis(tx: DataFrame) -> DataFrame:
    purch = tx.filter(~F.col("is_return"))
    return (
        purch.withColumn("month", F.trunc("invoice_day", "month"))
        .groupBy("month")
        .agg(F.sum("revenue").alias("revenue"),
             F.countDistinct("invoice").alias("orders"),
             F.countDistinct("customer_id").alias("active_customers"))
        .withColumn("avg_order_value", F.col("revenue") / F.col("orders"))
        .orderBy("month")
    )


# ----------------------------------------------------------------------------- main
def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    path = ensure_csv()
    log(f"Reading {path.name} with Spark")
    raw = load_raw(spark, path)

    log("Cleaning:")
    tx, steps = clean(raw)
    tx = tx.cache()

    bounds = tx.agg(F.min("invoice_day").alias("lo"), F.max("invoice_day").alias("hi")).first()
    data_start, data_end = bounds["lo"], bounds["hi"]
    snapshot_date = data_end + dt.timedelta(days=1)
    train_cutoff = dt.date.fromisoformat(TRAIN_CUTOFF)
    test_cutoff = dt.date.fromisoformat(TEST_CUTOFF)
    if test_cutoff + dt.timedelta(days=CHURN_HORIZON_DAYS) > snapshot_date:
        raise ValueError("TEST_CUTOFF + horizon runs past the end of the data.")
    log(f"Data covers {data_start} -> {data_end}; snapshot date {snapshot_date}")

    summary = tx.agg(
        F.countDistinct("customer_id").alias("customers"),
        F.countDistinct("invoice").alias("invoices"),
        F.countDistinct("stock_code").alias("products"),
        F.countDistinct("country").alias("countries"),
        F.sum(F.when(~F.col("is_return"), F.col("revenue"))).alias("gross_revenue"),
    ).first().asDict()

    log("Building customer tables")
    snap = build_customer_features(tx, snapshot_date).toPandas()
    train = build_customer_features(tx, train_cutoff, CHURN_HORIZON_DAYS,
                                    ACTIVE_LOOKBACK_DAYS).toPandas()
    test = build_customer_features(tx, test_cutoff, CHURN_HORIZON_DAYS,
                                   ACTIVE_LOOKBACK_DAYS).toPandas()
    snap.to_csv(SNAPSHOT_CSV, index=False)
    train.to_csv(CHURN_TRAIN_CSV, index=False)
    test.to_csv(CHURN_TEST_CSV, index=False)
    log(f"  snapshot {len(snap):,} customers | train {len(train):,} "
        f"(churn {train.target_churned.mean():.1%}) | test {len(test):,} "
        f"(churn {test.target_churned.mean():.1%})")

    cohort_counts(tx).toPandas().to_csv(COHORT_CSV, index=False)
    monthly_kpis(tx).toPandas().to_csv(MONTHLY_KPI_CSV, index=False)

    save_json({
        "cleaning_steps": steps,
        "raw_rows": steps[0]["rows_after"],
        "clean_rows": steps[-1]["rows_after"],
        "data_start": data_start, "data_end": data_end,
        "snapshot_date": snapshot_date,
        "train_cutoff": train_cutoff, "test_cutoff": test_cutoff,
        "churn_horizon_days": CHURN_HORIZON_DAYS,
        "active_lookback_days": ACTIVE_LOOKBACK_DAYS,
        "train_customers": len(train), "test_customers": len(test),
        "train_churn_rate": float(train.target_churned.mean()),
        "test_churn_rate": float(test.target_churned.mean()),
        **summary,
    }, OUTPUTS / "etl_report.json")
    spark.stop()


if __name__ == "__main__":
    main()
