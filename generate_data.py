"""Generate synthetic data for DataPulse.

Writes a CSV named ``data_10m.csv`` with 10,000,000 rows by default. The
generation is done in chunks so memory usage stays low (a few hundred MB of
peak RAM regardless of total row count) and is vectorised with numpy for speed.

Usage:
    python generate_data.py                  # 10M rows -> data_10m.csv
    python generate_data.py --rows 1000000   # 1M rows (handy for quick tests)
    python generate_data.py --out sample.csv --rows 1000
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CATEGORIES = ["Network", "Security", "Application", "Database", "Other"]
REGIONS = ["North", "South", "East", "West", "Central"]


def generate(path: str, total_rows: int, chunk_size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    # Random timestamps spread across the last 5 years.
    end = datetime.now().replace(microsecond=0)
    start = end - timedelta(days=5 * 365)
    start_epoch = int(start.timestamp())
    span_seconds = int((end - start).total_seconds())

    categories = np.array(CATEGORIES)
    regions = np.array(REGIONS)

    written = 0
    first_chunk = True
    started = time.perf_counter()

    while written < total_rows:
        n = min(chunk_size, total_rows - written)

        ids = np.arange(written + 1, written + n + 1, dtype=np.int64)

        # Random second offsets -> datetime64, formatted as ISO strings.
        offsets = rng.integers(0, span_seconds, size=n, dtype=np.int64)
        timestamps = (start_epoch + offsets).astype("datetime64[s]")

        values = np.round(rng.uniform(0.0, 100.0, size=n), 2)

        cats = categories[rng.integers(0, len(categories), size=n)]
        regs = regions[rng.integers(0, len(regions), size=n)]

        frame = pd.DataFrame(
            {
                "id": ids,
                "timestamp": timestamps,
                "value": values,
                "category": cats,
                "region": regs,
            }
        )

        frame.to_csv(
            path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        written += n
        first_chunk = False
        pct = written / total_rows * 100
        print(f"  {written:,}/{total_rows:,} rows ({pct:5.1f}%)", end="\r", flush=True)

    elapsed = time.perf_counter() - started
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print()
    print(f"Done: wrote {written:,} rows to {path} "
          f"({size_mb:.1f} MB) in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DataPulse data.")
    parser.add_argument("--out", default="data_10m.csv", help="Output CSV path.")
    parser.add_argument("--rows", type=int, default=10_000_000, help="Number of rows.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="Rows generated/written per chunk.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()

    print(f"Generating {args.rows:,} rows -> {args.out}")
    generate(args.out, args.rows, args.chunk_size, args.seed)


if __name__ == "__main__":
    main()
