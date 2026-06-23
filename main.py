"""DataPulse API — a high-performance local analytics engine.

FastAPI + DuckDB. The 10M-row CSV is loaded into an in-memory DuckDB table
*once* at startup (rather than re-parsed on every request), so analytical
queries over the full dataset return in milliseconds.

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

import csv
import io
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Iterator, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Data file path comes from the environment so deployments can point at a
# smaller production dataset (e.g. ./data_100k.csv) without code changes.
# Defaults to the local 10M-row dev file.
DATA_FILE_PATH = os.getenv("DATAPULSE_DATA_FILE", "./data_10m.csv")
# The Parquet path defaults to the data file with a .parquet extension (so a
# data file of ./data_100k.csv yields ./data_100k.parquet), overridable on its
# own. This keeps CSV- and Parquet-mode artefacts paired per dataset.
PARQUET_FILE_PATH = os.getenv(
    "DATAPULSE_PARQUET_FILE",
    os.path.splitext(DATA_FILE_PATH)[0] + ".parquet",
)
TABLE_NAME = "data"

# Set by the --parquet CLI flag (see __main__) or the env var directly. In
# Parquet mode DuckDB queries the file from disk via a view instead of loading
# the whole CSV into RAM — that's the memory/startup win.
USE_PARQUET = os.getenv("DATAPULSE_USE_PARQUET") == "1"

# CORS allowed origins come from the environment (comma-separated) so the
# deployed frontend's URL can be whitelisted without code changes. Defaults to
# the local Vite/React dev servers.
_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:3000"
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("DATAPULSE_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

# Columns that are safe to reference in SQL identifiers (sort_by, etc.).
# Anything outside this allow-list is rejected, which is what protects the
# dynamically-built parts of queries from SQL injection.
SORTABLE_COLUMNS = {"id", "timestamp", "value", "category", "region"}
TIME_INTERVALS = {"hour", "day", "week", "month", "year"}

# Approximate seconds per bucket, finest -> coarsest. Used to cap how many
# points a value_over_time chart can produce: an hour granularity over a 5-year
# range is ~44k near-flat buckets — meaningless to read and heavy to render. We
# reject any interval that would exceed MAX_CHART_BUCKETS for the selected range
# and suggest the finest one that fits. (month/year are nominal averages, which
# is plenty precise for a guard.)
INTERVAL_SECONDS = {
    "hour": 3_600,
    "day": 86_400,
    "week": 604_800,
    "month": 2_592_000,    # ~30 days
    "year": 31_536_000,    # 365 days
}
INTERVAL_ORDER = ["hour", "day", "week", "month", "year"]
MAX_CHART_BUCKETS = 2_000

# Shared module-level state, populated at startup.
con: Optional[duckdb.DuckDBPyConnection] = None
data_loaded: bool = False


# The CSV is read with this exact expression in both modes, so the column
# order and types (notably timestamp -> TIMESTAMP) are identical whether the
# data is materialised in RAM (CSV mode) or written to Parquet. Keeping it in
# one place is what guarantees byte-for-byte identical query results.
_READ_CSV = (
    f"SELECT id, CAST(timestamp AS TIMESTAMP) AS timestamp, value, category, region "
    f"FROM read_csv_auto(?, header=true, timestampformat='%Y-%m-%d %H:%M:%S')"
)


def _load_csv_mode(connection: duckdb.DuckDBPyConnection) -> bool:
    """Load the CSV into an in-memory table (default mode)."""
    if not os.path.exists(DATA_FILE_PATH):
        print(f"[startup] ERROR: data file not found at {DATA_FILE_PATH!r}. "
              f"Run `python generate_data.py` first.")
        return False

    print(f"[startup] CSV mode: loading {DATA_FILE_PATH!r} into in-memory table "
          f"{TABLE_NAME!r} ...")
    connection.execute(
        f"CREATE OR REPLACE TABLE {TABLE_NAME} AS {_READ_CSV}",
        [DATA_FILE_PATH],
    )
    rows = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    print(f"[startup] Loaded {rows:,} rows into RAM.")
    return True


def _load_parquet_mode(connection: duckdb.DuckDBPyConnection) -> bool:
    """Query a Parquet file directly from disk via a view (low-memory mode).

    Converts the CSV to Parquet once (skipped if the file already exists), then
    exposes it as a view named ``data`` so every endpoint's SQL is unchanged.
    Unlike CSV mode the rows are not held in RAM — DuckDB streams them from the
    Parquet file per query.
    """
    if not os.path.exists(PARQUET_FILE_PATH):
        if not os.path.exists(DATA_FILE_PATH):
            print(f"[startup] ERROR: neither {PARQUET_FILE_PATH!r} nor "
                  f"{DATA_FILE_PATH!r} found. Run `python generate_data.py` first.")
            return False
        print(f"[startup] Parquet mode: converting {DATA_FILE_PATH!r} -> "
              f"{PARQUET_FILE_PATH!r} (one-time) ...")
        # COPY straight from the CSV read expression; the source rows are
        # streamed through, not materialised into a table.
        connection.execute(
            f"COPY ({_READ_CSV}) TO '{PARQUET_FILE_PATH}' (FORMAT PARQUET)",
            [DATA_FILE_PATH],
        )
        print(f"[startup] Conversion done.")
    else:
        print(f"[startup] Parquet mode: using existing {PARQUET_FILE_PATH!r}.")

    # A view keeps reads on disk (no full load into RAM).
    connection.execute(
        f"CREATE OR REPLACE VIEW {TABLE_NAME} AS "
        f"SELECT * FROM read_parquet('{PARQUET_FILE_PATH}')"
    )
    rows = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    print(f"[startup] Querying {rows:,} rows from disk (Parquet).")
    return True


def _load_data(connection: duckdb.DuckDBPyConnection) -> bool:
    """Dispatch to CSV or Parquet startup based on the configured mode."""
    return _load_parquet_mode(connection) if USE_PARQUET else _load_csv_mode(connection)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle (replaces deprecated on_event handlers)."""
    global con, data_loaded
    con = duckdb.connect(database=":memory:")
    data_loaded = _load_data(con)
    if data_loaded:
        print("[startup] DataPulse API ready.")
    yield
    if con is not None:
        con.close()
        print("[shutdown] DuckDB connection closed.")


app = FastAPI(title="DataPulse API", version="1.0.0", lifespan=lifespan)

# Allow the configured frontends (dev servers by default; the deployed origin
# in production) to call the API. See CORS_ALLOW_ORIGINS / DATAPULSE_CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cursor() -> duckdb.DuckDBPyConnection:
    """Return a per-request DuckDB cursor.

    FastAPI runs sync endpoints across a threadpool, so multiple requests can
    execute simultaneously. A single shared DuckDB connection is NOT safe for
    concurrent queries — overlapping calls clobber each other's result set
    (manifesting as ``fetchone() -> None`` / "too many values to unpack" /
    500s). ``con.cursor()`` hands back a lightweight, independent handle to the
    same in-memory database that is safe to use on its own thread.
    """
    if not data_loaded or con is None:
        raise HTTPException(
            status_code=404,
            detail=f"Data file {DATA_FILE_PATH!r} not found or failed to load. "
                   f"Run `python generate_data.py` and restart the server.",
        )
    return con.cursor()


def _build_filters(
    category: Optional[str],
    min_value: Optional[float],
    max_value: Optional[float],
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[str, list[Any]]:
    """Build a parameterised WHERE clause shared by query/chart endpoints."""
    clauses: list[str] = []
    params: list[Any] = []

    if category is not None:
        clauses.append("category = ?")
        params.append(category)
    if min_value is not None:
        clauses.append("value >= ?")
        params.append(min_value)
    if max_value is not None:
        clauses.append("value <= ?")
        params.append(max_value)
    if start_date is not None:
        clauses.append("timestamp >= ?")
        params.append(start_date)
    if end_date is not None:
        clauses.append("timestamp <= ?")
        params.append(end_date)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _order_clause(sort_by: Optional[str], sort_order: str) -> str:
    """Validate sort params and return an ORDER BY clause (or empty string).

    sort_by is checked against the column allow-list and sort_order against
    asc/desc, so the result is safe to interpolate into SQL. Shared by the
    query and export endpoints. Raises HTTPException(400) on invalid input.

    Always ends with ``id`` as a tiebreaker (id is unique) so the ordering is a
    *total* order. Without it, sorting on a low-cardinality column like `value`
    (~10k distinct values over 10M rows) leaves tied rows in storage-dependent
    order, which makes pagination unstable and makes CSV- vs Parquet-mode
    results differ. With it, results are deterministic and identical.
    """
    if sort_by is not None and sort_by not in SORTABLE_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort_by {sort_by!r}. Allowed: {sorted(SORTABLE_COLUMNS)}",
        )
    if sort_order.lower() not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="sort_order must be 'asc' or 'desc'.")
    if sort_by is None or sort_by == "id":
        return f" ORDER BY id {sort_order.upper() if sort_by == 'id' else 'ASC'}"
    return f" ORDER BY {sort_by} {sort_order.upper()}, id ASC"


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DataPulse API is running"}


@app.get("/data/summary")
def data_summary() -> dict[str, Any]:
    """High-level statistics over the full dataset."""
    cur = _cursor()

    total_rows, min_ts, max_ts, avg_value = cur.execute(
        f"""
        SELECT COUNT(*),
               MIN(timestamp),
               MAX(timestamp),
               AVG(value)
        FROM {TABLE_NAME}
        """
    ).fetchone()

    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk) ->
    # the column name is at index 1.
    columns = [row[1] for row in cur.execute(
        f"PRAGMA table_info('{TABLE_NAME}')"
    ).fetchall()]

    unique_categories = [r[0] for r in cur.execute(
        f"SELECT DISTINCT category FROM {TABLE_NAME} ORDER BY category"
    ).fetchall()]

    return {
        "total_rows": int(total_rows),
        "columns": columns,
        "min_timestamp": min_ts.isoformat() if min_ts else None,
        "max_timestamp": max_ts.isoformat() if max_ts else None,
        "avg_value": round(float(avg_value), 4) if avg_value is not None else None,
        "unique_categories": unique_categories,
    }


@app.get("/data/query")
def data_query(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    sort_by: Optional[str] = None,
    sort_order: str = Query("asc"),
    category: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, Any]:
    """Paginated, sortable, filterable rows."""
    cur = _cursor()
    order_clause = _order_clause(sort_by, sort_order)

    where, params = _build_filters(
        category, min_value, max_value, start_date, end_date
    )

    total_count = cur.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME}{where}", params
    ).fetchone()[0]

    offset = (page - 1) * page_size
    sql = (
        f"SELECT * FROM {TABLE_NAME}{where}{order_clause} "
        f"LIMIT ? OFFSET ?"
    )
    cur.execute(sql, [*params, page_size, offset])
    columns = [d[0] for d in cur.description]
    rows = [dict(zip(columns, r)) for r in cur.fetchall()]

    # Make timestamps JSON-serialisable.
    for row in rows:
        ts = row.get("timestamp")
        if isinstance(ts, datetime):
            row["timestamp"] = ts.isoformat()

    return {
        "data": rows,
        "total_count": int(total_count),
        "current_page": page,
        "page_size": page_size,
    }


# Rows pulled from DuckDB per batch while streaming an export. Keeps the Python
# side bounded regardless of how many rows the filtered set contains.
EXPORT_BATCH = 50_000


def _stream_csv(sql: str, params: list[Any]) -> Iterator[str]:
    """Yield CSV text for ``sql`` in batches, never materialising it all.

    Uses its own cursor (the request thread's cursor must not be shared with a
    generator consumed during response streaming) and ``fetchmany`` so only
    EXPORT_BATCH rows are held in memory at a time.
    """
    cur = con.cursor()
    cur.execute(sql, params)

    header = io.StringIO()
    writer = csv.writer(header)
    writer.writerow([d[0] for d in cur.description])
    yield header.getvalue()

    while True:
        batch = cur.fetchmany(EXPORT_BATCH)
        if not batch:
            break
        chunk = io.StringIO()
        csv.writer(chunk).writerows(batch)
        yield chunk.getvalue()


@app.get("/data/export")
def data_export(
    sort_by: Optional[str] = None,
    sort_order: str = Query("asc"),
    category: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> StreamingResponse:
    """Stream the full filtered + sorted result set as CSV.

    Accepts the same filter/sort parameters as ``/data/query`` (minus
    pagination) so the export matches exactly what the table is showing. The
    response is streamed, so neither DuckDB nor FastAPI buffers the whole CSV.
    """
    # Validate up front so bad params return a clean 4xx (a generator can't set
    # the status code once streaming has started). _cursor() also guards 404.
    _cursor()
    order_clause = _order_clause(sort_by, sort_order)
    where, params = _build_filters(
        category, min_value, max_value, start_date, end_date
    )

    sql = f"SELECT * FROM {TABLE_NAME}{where}{order_clause}"
    return StreamingResponse(
        _stream_csv(sql, params),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="datapulse_export.csv"'
        },
    )


@app.get("/data/chart")
def data_chart(
    chart_type: str = Query(..., description="value_over_time | category_distribution"),
    interval: str = Query("day"),
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregated data shaped for charting."""
    cur = _cursor()

    where, params = _build_filters(
        category, None, None, start_date, end_date
    )

    if chart_type == "value_over_time":
        if interval not in TIME_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval {interval!r}. "
                       f"Allowed: {sorted(TIME_INTERVALS)}",
            )
        # Cap granularity to the selected range. Measure the actual span of the
        # (filtered) data and reject intervals that would over-produce buckets.
        min_ts, max_ts = cur.execute(
            f"SELECT MIN(timestamp), MAX(timestamp) FROM {TABLE_NAME}{where}",
            params,
        ).fetchone()
        if min_ts is not None and max_ts is not None:
            span = (max_ts - min_ts).total_seconds()
            est_buckets = span / INTERVAL_SECONDS[interval]
            if est_buckets > MAX_CHART_BUCKETS:
                suggested = next(
                    (iv for iv in INTERVAL_ORDER
                     if span / INTERVAL_SECONDS[iv] <= MAX_CHART_BUCKETS),
                    "year",
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Interval {interval!r} would produce ~{int(est_buckets):,} "
                        f"buckets over the selected range (max {MAX_CHART_BUCKETS}). "
                        f"Use a coarser interval (e.g. {suggested!r}) or narrow the "
                        f"date range."
                    ),
                )

        # interval is validated against the allow-list above.
        sql = (
            f"SELECT date_trunc('{interval}', timestamp) AS bucket, "
            f"       AVG(value) AS avg_value "
            f"FROM {TABLE_NAME}{where} "
            f"GROUP BY bucket ORDER BY bucket"
        )
        result = [
            {
                "time": bucket.isoformat() if isinstance(bucket, datetime) else str(bucket),
                "avg_value": round(float(avg), 4),
            }
            for bucket, avg in cur.execute(sql, params).fetchall()
        ]
        return {"chart_type": chart_type, "interval": interval, "data": result}

    if chart_type == "category_distribution":
        sql = (
            f"SELECT category, COUNT(*) AS count "
            f"FROM {TABLE_NAME}{where} "
            f"GROUP BY category ORDER BY count DESC"
        )
        result = [
            {"category": cat, "count": int(count)}
            for cat, count in cur.execute(sql, params).fetchall()
        ]
        return {"chart_type": chart_type, "data": result}

    raise HTTPException(
        status_code=400,
        detail=f"Invalid chart_type {chart_type!r}. "
               f"Allowed: ['value_over_time', 'category_distribution']",
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the DataPulse API.")
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Convert the CSV to Parquet once and query it from disk "
             "(lower memory, faster cold start) instead of loading into RAM.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # The lifespan reads this at startup; setting it here keeps `uvicorn
    # main:app` working too (export DATAPULSE_USE_PARQUET=1).
    if args.parquet:
        os.environ["DATAPULSE_USE_PARQUET"] = "1"
        USE_PARQUET = True

    uvicorn.run(app, host=args.host, port=args.port)
