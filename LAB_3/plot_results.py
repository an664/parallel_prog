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
for size, group in data.groupby("size"):
    group = group.sort_values("processes")
    plt.plot(group["processes"], group["time_sec"], marker="o", label=f"N={size}")
plt.title("MPI matrix multiplication")
plt.xlabel("Processes")
plt.ylabel("Time, seconds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "time_vs_processes.png", dpi=160)

baseline = data[data["processes"] == data["processes"].min()][["size", "time_sec"]]
baseline = baseline.rename(columns={"time_sec": "baseline_sec"})
speedup = data.merge(baseline, on="size")
speedup["speedup"] = speedup["baseline_sec"] / speedup["time_sec"]

plt.figure(figsize=(8, 5))
for size, group in speedup.groupby("size"):
    group = group.sort_values("processes")
    plt.plot(group["processes"], group["speedup"], marker="o", label=f"N={size}")
plt.title("MPI speedup")
plt.xlabel("Processes")
plt.ylabel("Speedup")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "speedup_vs_processes.png", dpi=160)
