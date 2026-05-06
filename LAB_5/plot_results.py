#!/usr/bin/env python3
import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lab 5 plots.")
    parser.add_argument("--input", default=str(ROOT / "measured_results.csv"))
    parser.add_argument("--prefix", default="measured")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data["gflops"] = data.get("gflops", data["operations"] / data["time_sec"] / 1e9)

    plt.figure(figsize=(8, 5))
    for size, group in data.groupby("size"):
        group = group.sort_values("processes")
        plt.plot(group["processes"], group["time_sec"], marker="o", label=f"N={size}")
    plt.title("MPI on Sergey Korolev cluster")
    plt.xlabel("MPI processes")
    plt.ylabel("Median compute time, seconds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / f"{args.prefix}_time_vs_processes.png", dpi=160)

    baseline = data[data["processes"] == data["processes"].min()][["size", "time_sec"]]
    baseline = baseline.rename(columns={"time_sec": "baseline_sec"})
    speedup = data.merge(baseline, on="size")
    speedup["speedup"] = speedup["baseline_sec"] / speedup["time_sec"]
    speedup["efficiency"] = speedup["speedup"] / speedup["processes"]

    plt.figure(figsize=(8, 5))
    for size, group in speedup.groupby("size"):
        group = group.sort_values("processes")
        plt.plot(group["processes"], group["speedup"], marker="o", label=f"N={size}")
    plt.title("MPI speedup on cluster")
    plt.xlabel("MPI processes")
    plt.ylabel("Speedup over 1 process")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / f"{args.prefix}_speedup_vs_processes.png", dpi=160)

    plt.figure(figsize=(8, 5))
    for size, group in speedup.groupby("size"):
        group = group.sort_values("processes")
        plt.plot(group["processes"], group["efficiency"], marker="o", label=f"N={size}")
    plt.title("MPI parallel efficiency")
    plt.xlabel("MPI processes")
    plt.ylabel("Efficiency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / f"{args.prefix}_efficiency_vs_processes.png", dpi=160)

    plt.figure(figsize=(8, 5))
    for size, group in data.groupby("size"):
        group = group.sort_values("processes")
        plt.plot(group["processes"], group["gflops"], marker="o", label=f"N={size}")
    plt.title("MPI throughput on cluster")
    plt.xlabel("MPI processes")
    plt.ylabel("GFLOPS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / f"{args.prefix}_gflops_vs_processes.png", dpi=160)

    lab3_path = ROOT.parent / "LAB_3" / "results.csv"
    if lab3_path.exists():
        lab3 = pd.read_csv(lab3_path)
        local_best = lab3.loc[lab3.groupby("size")["time_sec"].idxmin()][["size", "time_sec"]]
        cluster_best = data.loc[data.groupby("size")["time_sec"].idxmin()][["size", "time_sec"]]
        comparison = local_best.rename(columns={"time_sec": "Local MPI"}).merge(
            cluster_best.rename(columns={"time_sec": "Cluster measured"}),
            on="size",
        )

        plt.figure(figsize=(8, 5))
        for column in ["Local MPI", "Cluster measured"]:
            plt.plot(comparison["size"], comparison[column], marker="o", label=column)
        plt.title("Local MPI and cluster measured")
        plt.xlabel("Matrix size N")
        plt.ylabel("Best median compute time, seconds")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ROOT / f"{args.prefix}_cluster_vs_local_mpi.png", dpi=160)


if __name__ == "__main__":
    main()
