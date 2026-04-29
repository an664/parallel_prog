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
plt.plot(data["size"], data["time_sec"], marker="o")
plt.title("Sequential matrix multiplication")
plt.xlabel("Matrix size N")
plt.ylabel("Time, seconds")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "time_vs_size.png", dpi=160)

base = data.iloc[0]
theoretical = base["time_sec"] * (data["size"] / base["size"]) ** 3

plt.figure(figsize=(8, 5))
plt.plot(data["size"], data["time_sec"], marker="o", label="Measured median")
plt.plot(data["size"], theoretical, linestyle="--", marker="s", label="O(N^3) estimate")
plt.title("Measured time and O(N^3) growth")
plt.xlabel("Matrix size N")
plt.ylabel("Time, seconds")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "complexity_growth.png", dpi=160)
