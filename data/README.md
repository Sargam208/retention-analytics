# Data

Put these three files in `data/raw/` (or run `python data/download_data.py`, which fetches Online Retail II and Hillstrom; Cookie Cats needs a manual Kaggle download).

| File | Dataset | Size | Source |
|---|---|---|---|
| `online_retail_II.xlsx` **or** `online_retail_II.csv` | UCI Online Retail II: UK online gift retailer, Dec 2009 – Dec 2011 | ~1.07M line items | https://archive.ics.uci.edu/dataset/502/online+retail+ii (xlsx) or the Kaggle CSV copy (search "Online Retail II UCI") |
| `cookie_cats.csv` | Cookie Cats mobile-game A/B test (gate at level 30 vs 40) | 90,189 players | Kaggle, search "cookie cats A/B testing" |
| `hillstrom.csv` | Kevin Hillstrom / MineThatData e-mail campaign (randomised) | 64,000 customers | https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html, or automatic via `pip install scikit-uplift` (step 5 calls `fetch_hillstrom()` if the file is missing) |

Expected columns:

- **Online Retail II:** `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country` (the older "Online Retail" naming `InvoiceNo / UnitPrice / CustomerID` also works)
- **Cookie Cats:** `userid, version, sum_gamerounds, retention_1, retention_7`
- **Hillstrom:** `recency, history_segment, history, mens, womens, zip_code, newbie, channel, segment, visit, conversion, spend`

The raw files are not committed (about 150 MB, and each dataset has its own licence). `data/raw/` is git-ignored.

`data/processed/` is written by step 1 (`src/01_spark_etl.py`) and is committed so the dashboard can be deployed without the raw data:

| File | Contents |
|---|---|
| `churn_train.csv` | Customer features at the 2011-03-13 cutoff plus the 90-day churn label |
| `churn_test.csv` | The same at the 2011-09-11 cutoff (out-of-time test set) |
| `customers_snapshot.csv` | Features at the end of the data, used for segmentation and live scoring |
| `monthly_kpis.csv` | Revenue, orders, active customers, and average order value per month |
| `cohort_counts.csv` | Active customers by first-purchase cohort and months since first purchase |
