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

lab1 = pd.read_csv(ROOT.parent / "LAB_1" / "results.csv")
lab2 = pd.read_csv(ROOT.parent / "LAB_2" / "results.csv")
lab3 = pd.read_csv(ROOT.parent / "LAB_3" / "results.csv")
best_openmp = lab2.loc[lab2.groupby("size")["time_sec"].idxmin()][["size", "time_sec"]]
best_mpi = lab3.loc[lab3.groupby("size")["time_sec"].idxmin()][["size", "time_sec"]]
best_opencl = data.loc[data.groupby("size")["time_ms"].idxmin()][["size", "time_ms"]]
best_opencl["time_sec"] = best_opencl["time_ms"] / 1000.0

comparison = lab1[["size", "time_sec"]].rename(columns={"time_sec": "Sequential"})
comparison = comparison.merge(best_openmp.rename(columns={"time_sec": "Best OpenMP"}), on="size")
comparison = comparison.merge(best_mpi.rename(columns={"time_sec": "Best MPI"}), on="size")
comparison = comparison.merge(best_opencl[["size", "time_sec"]].rename(columns={"time_sec": "Best OpenCL"}), on="size")

plt.figure(figsize=(8, 5))
for column in ["Sequential", "Best OpenMP", "Best MPI", "Best OpenCL"]:
    plt.plot(comparison["size"], comparison[column], marker="o", label=column)
plt.title("Sequential, OpenMP, MPI and OpenCL")
plt.xlabel("Matrix size N")
plt.ylabel("Time, seconds")
plt.yscale("log")
plt.grid(True, alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "all_methods_time_comparison.png", dpi=160)

gflops = comparison[["size"]].copy()
operations = 2 * comparison["size"] ** 3
for column in ["Sequential", "Best OpenMP", "Best MPI", "Best OpenCL"]:
    gflops[column] = operations / comparison[column] / 1e9

plt.figure(figsize=(8, 5))
for column in ["Sequential", "Best OpenMP", "Best MPI", "Best OpenCL"]:
    plt.plot(gflops["size"], gflops[column], marker="o", label=column)
plt.title("Method throughput")
plt.xlabel("Matrix size N")
plt.ylabel("GFLOPS")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "all_methods_gflops_comparison.png", dpi=160)

speedup = comparison[["size"]].copy()
speedup["Best OpenMP"] = comparison["Sequential"] / comparison["Best OpenMP"]
speedup["Best MPI"] = comparison["Sequential"] / comparison["Best MPI"]
speedup["Best OpenCL"] = comparison["Sequential"] / comparison["Best OpenCL"]

plt.figure(figsize=(8, 5))
for column in ["Best OpenMP", "Best MPI", "Best OpenCL"]:
    plt.plot(speedup["size"], speedup[column], marker="o", label=column)
plt.title("Speedup over sequential")
plt.xlabel("Matrix size N")
plt.ylabel("Speedup")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "all_methods_speedup_comparison.png", dpi=160)
