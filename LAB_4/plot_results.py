#!/usr/bin/env python3
import os
from pathlib import Path

cache_dir = Path("/tmp/parallel_prog_plot_cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "results.csv")

plt.figure(figsize=(8, 5))
for (mode, local_size), group in data.groupby(["mode", "local_size"]):
    group = group.sort_values("size")
    plt.plot(group["size"], group["time_ms"], marker="o", label=f"{mode}, local={local_size}")
plt.title("OpenCL matrix multiplication")
plt.xlabel("Matrix size N")
plt.ylabel("Time, milliseconds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "time_vs_opencl_config.png", dpi=160)
