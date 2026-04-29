#!/usr/bin/env python3
import argparse
import csv
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(f"{matrix.shape[0]}\n")
        np.savetxt(file, matrix, fmt="%.8f")


def parse_time(output: str) -> float:
    match = re.search(r"time_sec=([0-9.]+)", output)
    if not match:
        raise RuntimeError(f"Cannot parse execution time:\n{output}")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential matrix multiplication benchmark.")
    parser.add_argument("--binary", default=str(ROOT / "matrix_seq"))
    parser.add_argument("--sizes", default="200,400,800,1200,1600,2000")
    parser.add_argument("--output", default=str(ROOT / "results.csv"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    sizes = [int(item) for item in args.sizes.split(",") if item]
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    rows = []

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")

    with tempfile.TemporaryDirectory(prefix="lab1_") as tmp:
        tmp_path = Path(tmp)
        for size in sizes:
            rng = np.random.default_rng(args.seed + size)
            a = rng.uniform(-1.0, 1.0, size=(size, size))
            b = rng.uniform(-1.0, 1.0, size=(size, size))

            a_path = tmp_path / "A.txt"
            b_path = tmp_path / "B.txt"
            c_path = tmp_path / "C.txt"
            write_matrix(a_path, a)
            write_matrix(b_path, b)

            times = []
            max_error = 0.0
            expected = a @ b
            for repeat in range(1, args.repeats + 1):
                completed = subprocess.run(
                    [args.binary, str(a_path), str(b_path), str(c_path)],
                    check=True,
                    text=True,
                    capture_output=True,
                    env=env,
                )
                elapsed = parse_time(completed.stdout)
                c = np.loadtxt(c_path, skiprows=1).reshape((size, size))
                max_error = max(max_error, float(np.max(np.abs(c - expected))))
                times.append(elapsed)
                print(f"N={size}, repeat={repeat}: {elapsed:.6f}s")

            median_time = statistics.median(times)
            rows.append(
                {
                    "size": size,
                    "operations": int(2 * size**3),
                    "repeats": args.repeats,
                    "time_sec": median_time,
                    "time_median_sec": median_time,
                    "time_mean_sec": statistics.fmean(times),
                    "time_min_sec": min(times),
                    "time_std_sec": statistics.stdev(times) if len(times) > 1 else 0.0,
                    "max_abs_error": max_error,
                }
            )
            print(f"N={size}: median={median_time:.6f}s, max_abs_error={max_error:.3e}")

    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            lineterminator="\n",
            fieldnames=[
                "size",
                "operations",
                "repeats",
                "time_sec",
                "time_median_sec",
                "time_mean_sec",
                "time_min_sec",
                "time_std_sec",
                "max_abs_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
