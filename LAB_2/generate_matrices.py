#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(f"{matrix.shape[0]}\n")
        np.savetxt(file, matrix, fmt="%.8f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate two square matrices.")
    parser.add_argument("size", type=int)
    parser.add_argument("--a", default="A.txt")
    parser.add_argument("--b", default="B.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    write_matrix(Path(args.a), rng.uniform(-1.0, 1.0, size=(args.size, args.size)))
    write_matrix(Path(args.b), rng.uniform(-1.0, 1.0, size=(args.size, args.size)))


if __name__ == "__main__":
    main()
