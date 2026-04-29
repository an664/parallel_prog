#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
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
    match = re.search(r"time_ms=([0-9.]+)", output)
    if not match:
        raise RuntimeError(f"Cannot parse execution time:\n{output}")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CUDA matrix multiplication benchmark.")
    parser.add_argument("--binary", default=str(ROOT / "matrix_cuda"))
    parser.add_argument("--sizes", default="256,512,1024,1600,2000")
    parser.add_argument("--blocks", default="8,16,32")
    parser.add_argument("--modes", default="naive,tiled")
    parser.add_argument("--output", default=str(ROOT / "results.csv"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if shutil.which("nvcc") is None and not Path(args.binary).exists():
        raise SystemExit("CUDA benchmark requires nvcc or an already built matrix_cuda binary.")

    sizes = [int(item) for item in args.sizes.split(",") if item]
    blocks = [int(item) for item in args.blocks.split(",") if item]
    modes = [item for item in args.modes.split(",") if item]
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

            for block_size in blocks:
                for mode in modes:
                    completed = subprocess.run(
                        [args.binary, str(a_path), str(b_path), str(c_path), str(block_size), mode],
                        check=True,
                        text=True,
                        capture_output=True,
                    )
                    elapsed_ms = parse_time(completed.stdout)
                    c = np.loadtxt(c_path, skiprows=1).reshape((size, size))
                    max_error = float(np.max(np.abs(c - expected)))
                    rows.append(
                        {
                            "size": size,
                            "block_size": block_size,
                            "mode": mode,
                            "time_ms": elapsed_ms,
                            "max_abs_error": max_error,
                        }
                    )
                    print(
                        f"N={size}, block={block_size}, mode={mode}: "
                        f"{elapsed_ms:.3f}ms, max_abs_error={max_error:.3e}"
                    )

    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["size", "block_size", "mode", "time_ms", "max_abs_error"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
