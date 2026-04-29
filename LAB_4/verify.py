#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def read_matrix(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        n = int(file.readline())
        data = np.loadtxt(file)
    return data.reshape((n, n))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CUDA matrix multiplication with NumPy.")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("c")
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    a = read_matrix(Path(args.a))
    b = read_matrix(Path(args.b))
    c = read_matrix(Path(args.c))
    expected = a @ b

    if np.allclose(c, expected, atol=args.atol):
        max_error = np.max(np.abs(c - expected))
        print(f"OK: result matches NumPy, max_abs_error={max_error:.3e}")
        return

    max_error = np.max(np.abs(c - expected))
    raise SystemExit(f"FAILED: max_abs_error={max_error:.3e}")


if __name__ == "__main__":
    main()
