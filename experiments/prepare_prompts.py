#!/usr/bin/env python3
"""
Prepare the first N prompts from MovieGenVideoBench_extended.txt.

Outputs:
1) A plain text prompt file (one prompt per line) for inference input
2) A CSV file with header: index,prompt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("prompts/MovieGenVideoBench_extended.txt"),
        help="Source prompt text file",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=25,
        help="Number of prompts to keep from the top of source file",
    )
    parser.add_argument(
        "--out-txt",
        type=Path,
        default=Path("experiments/prompts_25.txt"),
        help="Output prompt file used by inference.py",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("experiments/prompts.csv"),
        help="Output CSV file with index,prompt",
    )
    args = parser.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Missing source prompts: {args.src}")

    lines = [line.rstrip("\n") for line in args.src.read_text(encoding="utf-8").splitlines()]
    prompts = lines[: args.num]
    if len(prompts) < args.num:
        raise ValueError(
            f"Source has only {len(prompts)} prompts, but --num={args.num} was requested."
        )

    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    args.out_txt.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "prompt"])
        for idx, prompt in enumerate(prompts):
            writer.writerow([idx, prompt])

    print(f"Wrote {len(prompts)} prompts to {args.out_txt}")
    print(f"Wrote CSV to {args.out_csv}")


if __name__ == "__main__":
    main()
