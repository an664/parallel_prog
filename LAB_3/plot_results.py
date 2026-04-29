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

lab1 = pd.read_csv(ROOT.parent / "LAB_1" / "results.csv")
lab2 = pd.read_csv(ROOT.parent / "LAB_2" / "results.csv")
best_openmp = lab2.loc[lab2.groupby("size")["time_sec"].idxmin()][["size", "time_sec"]]
best_mpi = data.loc[data.groupby("size")["time_sec"].idxmin()][["size", "time_sec"]]

comparison = lab1[["size", "time_sec"]].rename(columns={"time_sec": "Sequential"})
comparison = comparison.merge(best_openmp.rename(columns={"time_sec": "Best OpenMP"}), on="size")
comparison = comparison.merge(best_mpi.rename(columns={"time_sec": "Best MPI"}), on="size")

plt.figure(figsize=(8, 5))
for column in ["Sequential", "Best OpenMP", "Best MPI"]:
    plt.plot(comparison["size"], comparison[column], marker="o", label=column)
plt.title("CPU method comparison")
plt.xlabel("Matrix size N")
plt.ylabel("Time, seconds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "cpu_methods_comparison.png", dpi=160)

speedup = comparison[["size"]].copy()
speedup["Best OpenMP"] = comparison["Sequential"] / comparison["Best OpenMP"]
speedup["Best MPI"] = comparison["Sequential"] / comparison["Best MPI"]

plt.figure(figsize=(8, 5))
for column in ["Best OpenMP", "Best MPI"]:
    plt.plot(speedup["size"], speedup[column], marker="o", label=column)
plt.title("CPU speedup over sequential")
plt.xlabel("Matrix size N")
plt.ylabel("Speedup")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "cpu_speedup_comparison.png", dpi=160)
