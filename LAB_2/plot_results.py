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
    group = group.sort_values("threads")
    plt.plot(group["threads"], group["time_sec"], marker="o", label=f"N={size}")
plt.title("OpenMP matrix multiplication")
plt.xlabel("Threads")
plt.ylabel("Time, seconds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "time_vs_threads.png", dpi=160)

baseline = data[data["threads"] == data["threads"].min()][["size", "time_sec"]]
baseline = baseline.rename(columns={"time_sec": "baseline_sec"})
speedup = data.merge(baseline, on="size")
speedup["speedup"] = speedup["baseline_sec"] / speedup["time_sec"]

plt.figure(figsize=(8, 5))
for size, group in speedup.groupby("size"):
    group = group.sort_values("threads")
    plt.plot(group["threads"], group["speedup"], marker="o", label=f"N={size}")
plt.title("OpenMP speedup")
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "speedup_vs_threads.png", dpi=160)

lab1 = pd.read_csv(ROOT.parent / "LAB_1" / "results.csv")
best_openmp = data.loc[data.groupby("size")["time_sec"].idxmin()][["size", "time_sec", "threads"]]
comparison = lab1[["size", "time_sec"]].rename(columns={"time_sec": "sequential_sec"})
comparison = comparison.merge(best_openmp.rename(columns={"time_sec": "openmp_best_sec"}), on="size")
comparison["speedup"] = comparison["sequential_sec"] / comparison["openmp_best_sec"]

plt.figure(figsize=(8, 5))
plt.plot(comparison["size"], comparison["sequential_sec"], marker="o", label="Sequential")
plt.plot(comparison["size"], comparison["openmp_best_sec"], marker="o", label="Best OpenMP")
plt.title("Sequential vs best OpenMP")
plt.xlabel("Matrix size N")
plt.ylabel("Time, seconds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "comparison_with_sequential.png", dpi=160)

plt.figure(figsize=(8, 5))
plt.plot(comparison["size"], comparison["speedup"], marker="o")
plt.title("Best OpenMP speedup over sequential")
plt.xlabel("Matrix size N")
plt.ylabel("Speedup")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "speedup_over_sequential.png", dpi=160)
