"""Small shared helpers (logging, JSON, Spark session, plotting defaults)."""
import json
import time
from pathlib import Path

import numpy as np

PALETTE = ["#1F6F8B", "#E3A33B", "#5E8C61", "#C8553D", "#6C5B7B", "#8A9BA8", "#2E4057"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, cls=_NpEncoder))
    log(f"saved {path}")


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def get_spark(app_name: str = "retention-analytics", driver_memory: str = "4g"):
    """Local Spark session. Settings chosen to behave the same on Spark 3.5 and 4.x."""
    import os
    import sys

    from pyspark.sql import SparkSession

    # Make Spark use this same Python (avoids "Python worker failed to connect" on Windows)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        # Bad casts / unparseable dates -> NULL instead of an exception (ANSI is on by
        # default in Spark 4), and use the modern date parser consistently.
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .getOrCreate()
    )


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "font.size": 10,
    })
    return plt
