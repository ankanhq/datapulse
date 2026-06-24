"""DataPulse API — a free, self-serve CSV analytics engine.

FastAPI + DuckDB. Any visitor uploads a CSV; it is parsed into an isolated,
per-dataset in-memory DuckDB table and explored through generic endpoints that
adapt to whatever columns the file has. Nothing is persisted to disk: uploads
live in memory only, keyed by an opaque dataset id, and are evicted once the
process is holding too many or they go stale — which keeps memory bounded on a
free-tier host and means one visitor never sees another's data.

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterator, Optional

import duckdb
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration (environment-driven so deployments need no code changes).
# ---------------------------------------------------------------------------

# Source CSV used by the "Try with sample data" button. The Dockerfile builds a
# 100k-row file here; locally it defaults to the dev dataset.
SAMPLE_FILE_PATH = os.getenv("DATAPULSE_DATA_FILE", "./data_10m.csv")

# CORS allowed origins (comma-separated). Defaults to the local dev servers; in
# production set DATAPULSE_CORS_ORIGINS to the deployed frontend's URL.
_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:3000"
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.getenv("DATAPULSE_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if o.strip()
]

# Free-tier guard rails.
MAX_UPLOAD_BYTES = int(os.getenv("DATAPULSE_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_DATASETS = int(os.getenv("DATAPULSE_MAX_DATASETS", "8"))      # LRU cap
DATASET_TTL_SECONDS = int(os.getenv("DATAPULSE_DATASET_TTL", "1800"))  # 30 min
# The sample is capped so it loads instantly and stays light regardless of how
# big the source file is.
SAMPLE_ROW_CAP = int(os.getenv("DATAPULSE_SAMPLE_ROWS", "50000"))

# Display/serving caps.
MAX_CHART_BUCKETS = 2_000      # time-series granularity guard
CATEGORY_TOP_N = 50            # category_counts slices
HISTOGRAM_BINS = 30            # numeric_histogram bins
EXPORT_BATCH = 50_000          # rows pulled per batch while streaming export

# Filter operators allowed per classified column type.
_OPS_BY_TYPE = {
    "number": {"eq", "neq", "gte", "lte"},
    "date": {"gte", "lte"},
    "text": {"eq", "neq", "contains"},
}
_SQL_OP = {"eq": "=", "neq": "!=", "gte": ">=", "lte": "<="}

# Approximate seconds per time-series bucket, finest -> coarsest.
INTERVAL_SECONDS = {
    "hour": 3_600,
    "day": 86_400,
    "week": 604_800,
    "month": 2_592_000,
    "year": 31_536_000,
}
INTERVAL_ORDER = ["hour", "day", "week", "month", "year"]


# ---------------------------------------------------------------------------
# Dataset registry (in-memory, thread-safe, evicting).
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    id: str
    table: str
    columns: list[dict[str, str]]   # [{name, type, sql_type}]
    row_count: int
    source: str                     # "upload" | "sample"
    name: str
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    @property
    def column_names(self) -> list[str]:
        return [c["name"] for c in self.columns]

    def type_of(self, name: str) -> Optional[str]:
        for c in self.columns:
            if c["name"] == name:
                return c["type"]
        return None

    def public(self) -> dict[str, Any]:
        """Shape returned to the client (no internal table name leaked)."""
        return {
            "dataset_id": self.id,
            "name": self.name,
            "source": self.source,
            "row_count": self.row_count,
            "columns": self.columns,
        }


con: Optional[duckdb.DuckDBPyConnection] = None
_registry: dict[str, Dataset] = {}
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global con
    con = duckdb.connect(database=":memory:")
    print(f"[startup] DataPulse ready. Sample source: {SAMPLE_FILE_PATH!r}. "
          f"Limits: {MAX_UPLOAD_BYTES // (1024*1024)}MB/file, "
          f"{MAX_DATASETS} datasets, {DATASET_TTL_SECONDS}s TTL.")
    yield
    if con is not None:
        con.close()
        print("[shutdown] DuckDB connection closed.")


app = FastAPI(title="DataPulse API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _qi(identifier: str) -> str:
    """Quote a SQL identifier (table/column), escaping embedded quotes.

    Column names come from user CSVs, so every identifier interpolated into SQL
    goes through here. Combined with validating column names against the
    dataset's known set, this is what keeps the dynamic SQL non-injectable.
    """
    return '"' + identifier.replace('"', '""') + '"'


def _classify(sql_type: str) -> str:
    """Map a DuckDB SQL type to one of: number | date | text."""
    t = sql_type.upper()
    if any(k in t for k in ("INT", "DECIMAL", "DOUBLE", "FLOAT", "REAL", "NUMERIC", "HUGEINT")):
        return "number"
    if any(k in t for k in ("TIMESTAMP", "DATE", "TIME")):
        return "date"
    return "text"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _fmt_num(x: float) -> str:
    """Human-readable number for chart labels (no scientific notation)."""
    if x == int(x):
        return f"{int(x):,}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _evict_locked() -> None:
    """Drop stale and excess datasets. Caller must hold _lock."""
    assert con is not None
    now = time.time()
    expired = [k for k, d in _registry.items() if now - d.last_access > DATASET_TTL_SECONDS]
    for k in expired:
        con.execute(f"DROP TABLE IF EXISTS {_qi(_registry[k].table)}")
        del _registry[k]
    # LRU: drop the least-recently-accessed until under the cap.
    while len(_registry) > MAX_DATASETS:
        oldest = min(_registry, key=lambda k: _registry[k].last_access)
        con.execute(f"DROP TABLE IF EXISTS {_qi(_registry[oldest].table)}")
        del _registry[oldest]


def _get_dataset(dataset_id: str) -> Dataset:
    """Fetch a dataset by id, bumping its access time; 404 if missing/expired."""
    with _lock:
        ds = _registry.get(dataset_id)
        if ds is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found. It may have expired — please upload your "
                       "CSV again or load the sample data.",
            )
        if time.time() - ds.last_access > DATASET_TTL_SECONDS:
            assert con is not None
            con.execute(f"DROP TABLE IF EXISTS {_qi(ds.table)}")
            del _registry[dataset_id]
            raise HTTPException(
                status_code=404,
                detail="Dataset expired. Please upload your CSV again or load the "
                       "sample data.",
            )
        ds.last_access = time.time()
        return ds


def _create_dataset(path: str, name: str, source: str, row_cap: Optional[int] = None) -> Dataset:
    """Load a CSV file at ``path`` into a fresh isolated table and register it.

    Raises HTTPException(400) with a friendly message on anything DuckDB can't
    parse (non-CSV, malformed, no header, empty). Caller is responsible for the
    file's lifetime — the data lives in the table afterwards, not on disk.
    """
    assert con is not None
    dataset_id = uuid.uuid4().hex
    table = f"ds_{dataset_id}"  # hex-only -> always a safe identifier
    limit = f" LIMIT {int(row_cap)}" if row_cap else ""

    with _lock:
        try:
            # Tolerant read: skip bad lines and pad ragged rows rather than
            # crashing, but still surface a clean error if nothing usable comes
            # out. sample_size=-1 scans the whole (<=25MB) file for type detection.
            con.execute(
                f"CREATE TABLE {_qi(table)} AS "
                f"SELECT * FROM read_csv_auto(?, header=true, ignore_errors=true, "
                f"null_padding=true, sample_size=-1){limit}",
                [path],
            )
        except Exception as exc:  # noqa: BLE001 - any parse failure -> friendly 400
            con.execute(f"DROP TABLE IF EXISTS {_qi(table)}")
            raise HTTPException(
                status_code=400,
                detail=f"Could not read this file as CSV. {str(exc).splitlines()[0]}",
            ) from exc

        info = con.execute(f"PRAGMA table_info({_qi(table)})").fetchall()
        # PRAGMA table_info -> (cid, name, type, notnull, dflt_value, pk)
        columns = [
            {"name": r[1], "type": _classify(r[2]), "sql_type": r[2]}
            for r in info
        ]
        row_count = con.execute(f"SELECT COUNT(*) FROM {_qi(table)}").fetchone()[0]

        # Validation that needs the parsed shape.
        problem: Optional[str] = None
        if not columns:
            problem = "No columns were detected in the file."
        elif row_count == 0:
            problem = "The file has no data rows. Make sure the first row is a header and there is at least one data row below it."
        elif all(c["name"].startswith("column") and c["name"][6:].isdigit() for c in columns):
            problem = "No header row was detected. Please include column names in the first row of the CSV."
        if problem:
            con.execute(f"DROP TABLE IF EXISTS {_qi(table)}")
            raise HTTPException(status_code=400, detail=problem)

        ds = Dataset(
            id=dataset_id, table=table, columns=columns,
            row_count=int(row_count), source=source, name=name,
        )
        _registry[dataset_id] = ds
        _evict_locked()
        return ds


def _cursor() -> duckdb.DuckDBPyConnection:
    """A per-request DuckDB cursor (safe for concurrent queries across threads)."""
    if con is None:
        raise HTTPException(status_code=503, detail="Server not ready.")
    return con.cursor()


def _build_where(ds: Dataset, filters_json: Optional[str]) -> tuple[str, list[Any]]:
    """Translate the JSON ``filters`` param into a parameterised WHERE clause.

    ``filters`` is a JSON array of {col, op, value}. Columns are validated
    against the dataset's schema and operators against an allow-list per column
    type; values are always bound, so the result is not injectable.
    """
    if not filters_json:
        return "", []
    try:
        items = json.loads(filters_json)
        assert isinstance(items, list)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid filters parameter.") from exc

    clauses: list[str] = []
    params: list[Any] = []
    for item in items:
        col = item.get("col")
        op = item.get("op")
        value = item.get("value")
        if col is None or value is None or value == "":
            continue
        col_type = ds.type_of(col)
        if col_type is None:
            raise HTTPException(status_code=400, detail=f"Unknown column {col!r}.")
        if op not in _OPS_BY_TYPE[col_type]:
            raise HTTPException(
                status_code=400,
                detail=f"Operator {op!r} is not valid for {col_type} column {col!r}.",
            )
        ident = _qi(col)
        if op == "contains":
            clauses.append(f"CAST({ident} AS VARCHAR) ILIKE ?")
            params.append(f"%{value}%")
        else:
            sql_op = _SQL_OP[op]
            if col_type == "number":
                clauses.append(f"{ident} {sql_op} CAST(? AS DOUBLE)")
            elif col_type == "date":
                clauses.append(f"{ident} {sql_op} CAST(? AS TIMESTAMP)")
            else:
                clauses.append(f"{ident} {sql_op} ?")
            params.append(value)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _order_clause(ds: Dataset, sort_by: Optional[str], sort_order: str) -> str:
    """Validated ORDER BY. Uses DuckDB's rowid as a stable tiebreaker so
    pagination is deterministic even when sorting a low-cardinality column."""
    if sort_order.lower() not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="sort_order must be 'asc' or 'desc'.")
    if sort_by is None or sort_by == "":
        return " ORDER BY rowid"
    if sort_by not in ds.column_names:
        raise HTTPException(status_code=400, detail=f"Unknown sort column {sort_by!r}.")
    return f" ORDER BY {_qi(sort_by)} {sort_order.upper()}, rowid"


# ---------------------------------------------------------------------------
# Endpoints.
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, Any]:
    return {"message": "DataPulse API is running", "active_datasets": len(_registry)}


def _excel_to_csv(src_path: str) -> str:
    """Read the first sheet of an Excel file and write it to a temp CSV path.

    By converting to CSV and then feeding it through the same read_csv_auto
    path as a normal upload, Excel files get byte-for-byte identical column
    detection, table, chart and export behaviour. pandas is imported lazily so
    the dependency is only loaded when an Excel file is actually uploaded.
    """
    import pandas as pd  # lazy: keeps base memory/startup low for the CSV path

    try:
        # sheet_name=0 -> first sheet. Engine is chosen from the file extension
        # (openpyxl for .xlsx, xlrd for legacy .xls).
        df = pd.read_excel(src_path, sheet_name=0)
    except Exception as exc:  # noqa: BLE001 - any read failure -> friendly 400
        raise HTTPException(
            status_code=400,
            detail=f"Could not read this Excel file. {str(exc).splitlines()[0]}",
        ) from exc

    csv_tmp = tempfile.NamedTemporaryFile(prefix="datapulse_", suffix=".csv", delete=False)
    csv_tmp.close()
    df.to_csv(csv_tmp.name, index=False)
    return csv_tmp.name


@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Receive a CSV or Excel upload, load it into an isolated table, return id + schema."""
    filename = file.filename or "upload.csv"
    lower = filename.lower()
    is_excel = lower.endswith((".xlsx", ".xls"))
    if not (is_excel or lower.endswith((".csv", ".tsv", ".txt"))):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV or Excel file (.csv, .xlsx, .xls).",
        )

    # Stream to a temp file, enforcing the size cap as we go (never trust the
    # client's Content-Length). Temp files are deleted right after loading.
    suffix = os.path.splitext(filename)[1] or ".csv"
    tmp = tempfile.NamedTemporaryFile(prefix="datapulse_", suffix=suffix, delete=False)
    cleanup = [tmp.name]
    size = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File is too large. The limit is "
                           f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            tmp.write(chunk)
        tmp.close()

        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        if is_excel:
            load_path = _excel_to_csv(tmp.name)
            cleanup.append(load_path)
        else:
            load_path = tmp.name

        ds = _create_dataset(load_path, name=filename, source="upload")
        return ds.public()
    finally:
        for path in cleanup:
            try:
                os.unlink(path)
            except OSError:
                pass


class PasteRequest(BaseModel):
    text: str
    name: Optional[str] = None


@app.post("/datasets/text")
def upload_text(req: PasteRequest) -> dict[str, Any]:
    """Load pasted CSV / tab-separated text, exactly like an uploaded file.

    DuckDB's CSV reader auto-detects the delimiter, so spreadsheet copy-paste
    (tab-separated) and comma-separated text both work. Same size limit and
    validation as the file upload path.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text was provided to analyze.")
    data = req.text.encode("utf-8")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Pasted text is too large. The limit is "
                   f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    tmp = tempfile.NamedTemporaryFile(prefix="datapulse_", suffix=".csv", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        ds = _create_dataset(tmp.name, name=(req.name or "Pasted data"), source="paste")
        return ds.public()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.post("/datasets/sample")
def create_sample_dataset() -> dict[str, Any]:
    """Load the bundled demo dataset so first-time visitors see it work instantly."""
    if not os.path.exists(SAMPLE_FILE_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Sample data is unavailable on this server "
                   f"({SAMPLE_FILE_PATH!r} not found).",
        )
    ds = _create_dataset(
        SAMPLE_FILE_PATH, name="Sample data", source="sample", row_cap=SAMPLE_ROW_CAP
    )
    return ds.public()


@app.get("/datasets/{dataset_id}/summary")
def dataset_summary(dataset_id: str) -> dict[str, Any]:
    """Row/column counts plus per-column basic stats (adapts to column types)."""
    ds = _get_dataset(dataset_id)
    cur = _cursor()

    # One scan computes every column's stats. Build the SELECT list and remember
    # which result slot maps to which (column, stat).
    select_parts: list[str] = ["COUNT(*)"]
    plan: list[tuple[str, str]] = []  # (column_name, stat_key) aligned to slots after COUNT(*)
    for c in ds.columns:
        ident = _qi(c["name"])
        if c["type"] == "number":
            select_parts += [f"MIN({ident})", f"MAX({ident})", f"AVG({ident})"]
            plan += [(c["name"], "min"), (c["name"], "max"), (c["name"], "avg")]
        elif c["type"] == "date":
            select_parts += [f"MIN({ident})", f"MAX({ident})"]
            plan += [(c["name"], "min"), (c["name"], "max")]
        else:
            select_parts += [f"COUNT(DISTINCT {ident})"]
            plan += [(c["name"], "distinct")]

    row = cur.execute(
        f"SELECT {', '.join(select_parts)} FROM {_qi(ds.table)}"
    ).fetchone()

    total_rows = int(row[0])
    stats: dict[str, dict[str, Any]] = {c["name"]: {"type": c["type"]} for c in ds.columns}
    for slot, (col, key) in enumerate(plan, start=1):
        val = row[slot]
        if key == "avg" and val is not None:
            val = round(float(val), 4)
        stats[col][key] = _jsonable(val)

    return {
        "total_rows": total_rows,
        "total_columns": len(ds.columns),
        "columns": [{"name": c["name"], **stats[c["name"]]} for c in ds.columns],
    }


@app.get("/datasets/{dataset_id}/query")
def dataset_query(
    dataset_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    sort_by: Optional[str] = None,
    sort_order: str = Query("asc"),
    filters: Optional[str] = None,
) -> dict[str, Any]:
    """Paginated, sortable, filterable rows over the uploaded dataset."""
    ds = _get_dataset(dataset_id)
    cur = _cursor()

    where, params = _build_where(ds, filters)
    order = _order_clause(ds, sort_by, sort_order)

    total_count = cur.execute(
        f"SELECT COUNT(*) FROM {_qi(ds.table)}{where}", params
    ).fetchone()[0]

    offset = (page - 1) * page_size
    cur.execute(
        f"SELECT * FROM {_qi(ds.table)}{where}{order} LIMIT ? OFFSET ?",
        [*params, page_size, offset],
    )
    cols = [d[0] for d in cur.description]
    rows = [{k: _jsonable(v) for k, v in zip(cols, r)} for r in cur.fetchall()]

    return {
        "data": rows,
        "columns": ds.columns,
        "total_count": int(total_count),
        "current_page": page,
        "page_size": page_size,
    }


def _stream_csv(sql: str, params: list[Any]) -> Iterator[str]:
    """Yield CSV text for ``sql`` in batches (own cursor, never fully buffered)."""
    cur = con.cursor()
    cur.execute(sql, params)
    header = io.StringIO()
    csv.writer(header).writerow([d[0] for d in cur.description])
    yield header.getvalue()
    while True:
        batch = cur.fetchmany(EXPORT_BATCH)
        if not batch:
            break
        chunk = io.StringIO()
        csv.writer(chunk).writerows(batch)
        yield chunk.getvalue()


@app.get("/datasets/{dataset_id}/export")
def dataset_export(
    dataset_id: str,
    sort_by: Optional[str] = None,
    sort_order: str = Query("asc"),
    filters: Optional[str] = None,
) -> StreamingResponse:
    """Stream the current filtered + sorted view as CSV."""
    ds = _get_dataset(dataset_id)
    where, params = _build_where(ds, filters)
    order = _order_clause(ds, sort_by, sort_order)
    sql = f"SELECT * FROM {_qi(ds.table)}{where}{order}"
    return StreamingResponse(
        _stream_csv(sql, params),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="datapulse_export.csv"'},
    )


def _pick_interval(span_seconds: float, requested: Optional[str]) -> str:
    """Choose a time-series interval that stays under MAX_CHART_BUCKETS."""
    def fits(iv: str) -> bool:
        return span_seconds / INTERVAL_SECONDS[iv] <= MAX_CHART_BUCKETS
    if requested and requested in INTERVAL_SECONDS and fits(requested):
        return requested
    return next((iv for iv in INTERVAL_ORDER if fits(iv)), "year")


@app.get("/datasets/{dataset_id}/chart")
def dataset_chart(
    dataset_id: str,
    chart_type: str = Query(..., description="category_counts | time_series | numeric_histogram"),
    column: str = Query(..., description="Column to chart"),
    y_column: Optional[str] = None,
    agg: str = Query("count", description="count | avg | sum (time_series)"),
    interval: Optional[str] = None,
    filters: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregated data shaped for charting; adapts to the chosen column's type."""
    ds = _get_dataset(dataset_id)
    cur = _cursor()
    if column not in ds.column_names:
        raise HTTPException(status_code=400, detail=f"Unknown column {column!r}.")
    where, params = _build_where(ds, filters)
    col = _qi(column)
    tbl = _qi(ds.table)

    if chart_type == "category_counts":
        sql = (
            f"SELECT CAST({col} AS VARCHAR) AS label, COUNT(*) AS count "
            f"FROM {tbl}{where} GROUP BY label ORDER BY count DESC LIMIT {CATEGORY_TOP_N}"
        )
        data = [{"label": lbl, "count": int(cnt)} for lbl, cnt in cur.execute(sql, params).fetchall()]
        return {"chart_type": chart_type, "column": column, "data": data}

    if chart_type == "time_series":
        if ds.type_of(column) != "date":
            raise HTTPException(status_code=400, detail=f"{column!r} is not a date column.")
        if agg not in {"count", "avg", "sum"}:
            raise HTTPException(status_code=400, detail="agg must be count, avg or sum.")
        if agg in {"avg", "sum"}:
            if not y_column or y_column not in ds.column_names:
                raise HTTPException(status_code=400, detail="A numeric y_column is required for avg/sum.")
            if ds.type_of(y_column) != "number":
                raise HTTPException(status_code=400, detail=f"{y_column!r} is not a numeric column.")

        min_ts, max_ts = cur.execute(
            f"SELECT MIN({col}), MAX({col}) FROM {tbl}{where}", params
        ).fetchone()
        if min_ts is None or max_ts is None:
            return {"chart_type": chart_type, "column": column, "interval": interval, "data": []}
        span = max((max_ts - min_ts).total_seconds(), 1.0)
        chosen = _pick_interval(span, interval)

        if agg == "count":
            agg_expr = "COUNT(*)"
        else:
            agg_expr = f"{agg.upper()}({_qi(y_column)})"
        sql = (
            f"SELECT date_trunc('{chosen}', {col}) AS bucket, {agg_expr} AS value "
            f"FROM {tbl}{where} GROUP BY bucket ORDER BY bucket"
        )
        data = [
            {"time": _jsonable(bucket), "value": round(float(v), 4) if v is not None else None}
            for bucket, v in cur.execute(sql, params).fetchall()
        ]
        return {"chart_type": chart_type, "column": column, "interval": chosen,
                "agg": agg, "y_column": y_column, "data": data}

    if chart_type == "numeric_histogram":
        if ds.type_of(column) != "number":
            raise HTTPException(status_code=400, detail=f"{column!r} is not a numeric column.")
        lo, hi, n = cur.execute(
            f"SELECT MIN({col}), MAX({col}), COUNT({col}) FROM {tbl}{where}", params
        ).fetchone()
        if n == 0 or lo is None:
            return {"chart_type": chart_type, "column": column, "data": []}
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            return {"chart_type": chart_type, "column": column,
                    "data": [{"label": _fmt_num(lo), "bin_start": lo, "bin_end": lo, "count": int(n)}]}
        bins = HISTOGRAM_BINS
        width = (hi - lo) / bins
        # Bucket each value; clamp the max into the last bin. lo/hi/width are
        # server-computed floats, safe to inline.
        sql = (
            f"SELECT LEAST({bins - 1}, CAST(FLOOR(({col} - {lo}) / {width}) AS INTEGER)) AS b, "
            f"COUNT(*) AS count FROM {tbl}{where} WHERE {col} IS NOT NULL "
            f"GROUP BY b ORDER BY b"
        )
        counts = {int(b): int(c) for b, c in cur.execute(sql, params).fetchall()}
        data = []
        for b in range(bins):
            start = lo + b * width
            end = start + width
            data.append({
                "label": f"{_fmt_num(start)}–{_fmt_num(end)}",
                "bin_start": round(start, 6),
                "bin_end": round(end, 6),
                "count": counts.get(b, 0),
            })
        return {"chart_type": chart_type, "column": column, "data": data}

    raise HTTPException(
        status_code=400,
        detail=f"Invalid chart_type {chart_type!r}. Allowed: "
               f"category_counts, time_series, numeric_histogram.",
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the DataPulse API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
