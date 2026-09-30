"""Run the whole pipeline:  python run_all.py            (all steps)
                            python run_all.py --from 2   (skip the Spark ETL)
                            python run_all.py --only 4   (just the A/B test)"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

STEPS = {
    1: "01_spark_etl.py",
    2: "02_segmentation.py",
    3: "03_churn_model.py",
    4: "04_ab_test.py",
    5: "05_uplift.py",
    6: "06_make_report.py",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only", type=int)
    args = ap.parse_args()
    src = Path(__file__).parent / "src"
    todo = [args.only] if args.only else [k for k in STEPS if k >= args.start]
    for k in todo:
        print(f"\n{'=' * 70}\nSTEP {k}: {STEPS[k]}\n{'=' * 70}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, STEPS[k]], cwd=src)
        if r.returncode != 0:
            sys.exit(f"Step {k} failed.")
        print(f"-- step {k} done in {time.time() - t0:.0f}s")
    print("\nAll done. Results: outputs/REPORT.md | Dashboard: streamlit run app/app.py")
