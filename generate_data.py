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
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

CATEGORIES = ["Network", "Security", "Application", "Database", "Other"]
REGIONS = ["North", "South", "East", "West", "Central"]

# Product lines for the "story" sample. 'Alpha' is intentionally dominant so
# Evidence Mode surfaces a strong concentration insight.
PRODUCT_LINES = ["Alpha", "Bravo", "Charlie", "Delta"]


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


def generate_story(path: str, days: int, seed: int) -> None:
    """Write a small, business-shaped sample that tells a clear story.

    Unlike the uniform-random ``generate`` (a stress/scale dataset), this one is
    built so Evidence Mode shows strong, high-confidence cards on first click:

    * a **date** column spanning ``days`` days;
    * two numeric metrics (``units_sold``, ``revenue``) where revenue trends up
      over time and the daily row volume also ramps up (a confident trend);
    * ``units_sold`` and ``revenue`` move together (a strong correlation);
    * one **dominant category** — 'Alpha' is ~65% of ``product_line``;
    * one **obvious outlier** — a single row with a huge revenue spike.

    It stays a few thousand rows so it loads instantly.
    """
    rng = np.random.default_rng(seed)
    start = date.today() - timedelta(days=days)

    dates: list[str] = []
    product_lines: list[str] = []
    regions: list[str] = []
    units: list[int] = []
    revenue: list[float] = []

    regions_arr = np.array(REGIONS)
    others = np.array([p for p in PRODUCT_LINES if p != "Alpha"])

    for d in range(days):
        # Daily row volume ramps up smoothly over time -> high-R² trend card.
        n = max(1, int(round(6 + 0.20 * d + rng.normal(0, 1.2))))
        day_str = (start + timedelta(days=d)).isoformat()

        # 'Alpha' dominates (~65%); the rest split across the other lines.
        is_alpha = rng.random(n) < 0.65
        lines = np.where(is_alpha, "Alpha", others[rng.integers(0, len(others), size=n)])

        # Units are drawn uniformly (with a mild upward drift) so they have no
        # long tail and produce no IQR outliers of their own. Revenue rises on a
        # straight additive line over time (plus a units term), so it *clearly
        # trends up* with a wide, even spread — leaving the single planted spike
        # below as the one obvious anomaly in the whole dataset.
        u = np.round(rng.uniform(22, 38, size=n) + 0.04 * d).astype(int)
        rev = np.round(900 + 18.0 * d + 30.0 * (u - 30) + rng.normal(0, 200, size=n), 2)
        rev = np.maximum(rev, 50.0)

        dates.extend([day_str] * n)
        product_lines.extend(lines.tolist())
        regions.extend(regions_arr[rng.integers(0, len(regions_arr), size=n)].tolist())
        units.extend(u.tolist())
        revenue.extend(rev.tolist())

    frame = pd.DataFrame({
        "date": dates,
        "units_sold": units,
        "revenue": revenue,
        "product_line": product_lines,
        "region": regions,
    })

    # One obvious outlier: a single record with a huge revenue spike.
    spike_idx = int(rng.integers(len(frame) // 4, len(frame) // 2))
    frame.loc[spike_idx, "revenue"] = 500000.0

    frame.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Done: wrote {len(frame):,} rows to {path} "
          f"({size_mb:.2f} MB) across {days} days")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DataPulse data.")
    parser.add_argument("--out", default="data_10m.csv", help="Output CSV path.")
    parser.add_argument("--rows", type=int, default=10_000_000, help="Number of rows.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="Rows generated/written per chunk.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--profile", choices=["random", "story"], default="random",
                        help="'random' = uniform scale dataset; 'story' = the small, "
                             "business-shaped sample used by 'Try with sample data'.")
    parser.add_argument("--days", type=int, default=180,
                        help="Days spanned by the 'story' profile.")
    args = parser.parse_args()

    if args.profile == "story":
        print(f"Generating story sample over {args.days} days -> {args.out}")
        generate_story(args.out, args.days, args.seed)
        return

    print(f"Generating {args.rows:,} rows -> {args.out}")
    generate(args.out, args.rows, args.chunk_size, args.seed)


if __name__ == "__main__":
    main()
