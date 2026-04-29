#!/usr/bin/env python3
import argparse
import csv
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
    match = re.search(r"^time_ms=([0-9.]+)", output, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Cannot parse execution time:\n{output}")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenCL matrix multiplication benchmark.")
    parser.add_argument("--binary", default=str(ROOT / "matrix_opencl"))
    parser.add_argument("--sizes", default="200,400,800,1200,1600,2000")
    parser.add_argument("--local-sizes", default="4,8,16")
    parser.add_argument("--modes", default="naive,tiled")
    parser.add_argument("--output", default=str(ROOT / "results.csv"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    sizes = [int(item) for item in args.sizes.split(",") if item]
    local_sizes = [int(item) for item in args.local_sizes.split(",") if item]
    modes = [item for item in args.modes.split(",") if item]
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    rows = []

    with tempfile.TemporaryDirectory(prefix="lab4_") as tmp:
        tmp_path = Path(tmp)
        for size in sizes:
            rng = np.random.default_rng(args.seed + size)
            a = rng.uniform(-1.0, 1.0, size=(size, size))
            b = rng.uniform(-1.0, 1.0, size=(size, size))
            expected = a @ b

            a_path = tmp_path / "A.txt"
            b_path = tmp_path / "B.txt"
            c_path = tmp_path / "C.txt"
            write_matrix(a_path, a)
            write_matrix(b_path, b)

            for local_size in local_sizes:
                for mode in modes:
                    times = []
                    max_error = 0.0
                    for repeat in range(1, args.repeats + 1):
                        completed = subprocess.run(
                            [args.binary, str(a_path), str(b_path), str(c_path), str(local_size), mode],
                            check=True,
                            text=True,
                            capture_output=True,
                        )
                        elapsed_ms = parse_time(completed.stdout)
                        c = np.loadtxt(c_path, skiprows=1).reshape((size, size))
                        max_error = max(max_error, float(np.max(np.abs(c - expected))))
                        times.append(elapsed_ms)
                        print(
                            f"N={size}, local={local_size}, mode={mode}, "
                            f"repeat={repeat}: {elapsed_ms:.3f}ms"
                        )

                    median_time = statistics.median(times)
                    rows.append(
                        {
                            "size": size,
                            "local_size": local_size,
                            "mode": mode,
                            "repeats": args.repeats,
                            "time_ms": median_time,
                            "time_median_ms": median_time,
                            "time_mean_ms": statistics.fmean(times),
                            "time_min_ms": min(times),
                            "time_std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                            "max_abs_error": max_error,
                        }
                    )
                    print(
                        f"N={size}, local={local_size}, mode={mode}: "
                        f"median={median_time:.3f}ms, max_abs_error={max_error:.3e}"
                    )

    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            lineterminator="\n",
            fieldnames=[
                "size",
                "local_size",
                "mode",
                "repeats",
                "time_ms",
                "time_median_ms",
                "time_mean_ms",
                "time_min_ms",
                "time_std_ms",
                "max_abs_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
