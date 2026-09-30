"""Best-effort downloader. Cookie Cats needs Kaggle (manual or `kaggle` CLI).

    python data/download_data.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "raw"
RAW.mkdir(exist_ok=True)

UCI_ZIP = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
HILLSTROM = ("http://www.minethatdata.com/"
             "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def online_retail():
    if (RAW / "online_retail_II.xlsx").exists() or (RAW / "online_retail_II.csv").exists():
        return print("online retail: already present")
    try:
        z = zipfile.ZipFile(io.BytesIO(get(UCI_ZIP)))
        name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        (RAW / "online_retail_II.xlsx").write_bytes(z.read(name))
        print("online retail: downloaded")
    except Exception as e:
        print(f"online retail: FAILED ({e}). Download manually from "
              "https://archive.ics.uci.edu/dataset/502/online+retail+ii")


def hillstrom():
    if (RAW / "hillstrom.csv").exists():
        return print("hillstrom: already present")
    try:
        (RAW / "hillstrom.csv").write_bytes(get(HILLSTROM))
        print("hillstrom: downloaded")
    except Exception as e:
        print(f"hillstrom: direct download failed ({e}); step 5 will try scikit-uplift's "
              "fetch_hillstrom() automatically (pip install scikit-uplift).")


def cookie_cats():
    if (RAW / "cookie_cats.csv").exists():
        return print("cookie cats: already present")
    try:
        import kaggle  # needs ~/.kaggle/kaggle.json
        kaggle.api.dataset_download_files("mursideyarkin/mobile-games-ab-testing-cookie-cats",
                                          path=str(RAW), unzip=True)
        print("cookie cats: downloaded via Kaggle API")
    except Exception as e:
        print(f"cookie cats: please download cookie_cats.csv from Kaggle into {RAW} ({e})")


if __name__ == "__main__":
    online_retail(); hillstrom(); cookie_cats()
