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
import ipaddress
import json
import os
import re
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import duckdb
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import insights as insights_mod

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

# Directory for shared Evidence-Mode reports. NOTE: persistence here is
# best-effort — free hosts (Render free tier) use an ephemeral filesystem that is
# wiped on redeploy/restart, so shared links may expire. For durable sharing this
# would move to object storage or a database.
REPORTS_DIR = os.getenv("DATAPULSE_REPORTS_DIR", "./.reports")
MAX_REPORT_BYTES = 4 * 1024 * 1024  # guard against oversized report payloads

# Display/serving caps.
MAX_CHART_BUCKETS = 2_000      # time-series granularity guard
CATEGORY_TOP_N = 50            # category_counts slices
HISTOGRAM_BINS = 30            # numeric_histogram bins
SPLIT_SERIES_TOP_N = 8         # max lines when a time series is split by a category
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
    source: str                     # "upload" | "paste" | "sample" | "url"
    name: str
    source_url: Optional[str] = None  # set when source == "url" (re-fetchable)
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
            "source_url": self.source_url,
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


def _build_table_locked(table: str, path: str, row_cap: Optional[int]) -> tuple[list[dict], int]:
    """Load the CSV at ``path`` into ``table`` and validate the parsed shape.

    Assumes ``_lock`` is held. Returns ``(columns, row_count)`` or raises
    HTTPException(400) with a friendly message, dropping the table on failure.
    """
    assert con is not None
    limit = f" LIMIT {int(row_cap)}" if row_cap else ""
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

    return columns, int(row_count)


def _create_dataset(path: str, name: str, source: str, row_cap: Optional[int] = None,
                    source_url: Optional[str] = None) -> Dataset:
    """Load a CSV file at ``path`` into a fresh isolated table and register it.

    Raises HTTPException(400) with a friendly message on anything DuckDB can't
    parse (non-CSV, malformed, no header, empty). Caller is responsible for the
    file's lifetime — the data lives in the table afterwards, not on disk.
    """
    assert con is not None
    dataset_id = uuid.uuid4().hex
    table = f"ds_{dataset_id}"  # hex-only -> always a safe identifier

    with _lock:
        columns, row_count = _build_table_locked(table, path, row_cap)
        ds = Dataset(
            id=dataset_id, table=table, columns=columns,
            row_count=row_count, source=source, name=name, source_url=source_url,
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


def _augment_where(where: str, extra: str) -> str:
    """Append a server-built predicate to a WHERE clause from ``_build_where``."""
    return f"{where} AND {extra}" if where.strip() else f" WHERE {extra}"


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


# ---------------------------------------------------------------------------
# Fetching a CSV from a public URL (e.g. a Google Sheet "Publish to web" link).
# ---------------------------------------------------------------------------

URL_FETCH_TIMEOUT = int(os.getenv("DATAPULSE_URL_FETCH_TIMEOUT", "20"))  # seconds


_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _reject_if_private(host: str) -> None:
    """Resolve ``host`` and refuse private/loopback/link-local/reserved targets.

    This is the core SSRF guard: it stops a user-supplied URL from being used to
    reach the server's own network (e.g. cloud metadata at 169.254.169.254).

    NAT64 and IPv4-mapped IPv6 addresses are unwrapped to the real IPv4 they point
    at before the check, so a public IPv4 host reached over IPv6 (common on
    IPv6-only / NAT64 networks) isn't mis-flagged as "reserved" and blocked.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail=f"Could not resolve the host {host!r}.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if isinstance(ip, ipaddress.IPv6Address):
            if ip.ipv4_mapped:
                ip = ip.ipv4_mapped
            elif ip in _NAT64_PREFIX:
                ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise HTTPException(
                status_code=400,
                detail="That URL resolves to a private or local address, which isn't allowed.",
            )


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="The link must start with http:// or https://.")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="That doesn't look like a valid URL.")
    _reject_if_private(parsed.hostname)


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but re-validate every hop so a redirect can't be used to
    reach a private/internal address after the initial check passed."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_csv_url(url: str) -> str:
    """Fetch a public CSV URL server-side into a size-capped temp .csv file.

    Validates the target is public (http/https, not a private address) at every
    redirect hop and rejects obvious non-CSV (HTML) responses. Returns the temp
    path; the caller must delete it.
    """
    url = url.strip()
    _validate_public_url(url)
    opener = urllib.request.build_opener(_ValidatingRedirectHandler)
    req = urllib.request.Request(url, headers={"User-Agent": "DataPulse/1.0 (+csv-fetch)"})
    try:
        resp = opener.open(req, timeout=URL_FETCH_TIMEOUT)
    except HTTPException:
        raise  # a redirect hop failed validation
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"The link returned an error (HTTP {exc.code}). Make sure the sheet is "
                   "published to the web and publicly accessible.",
        ) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not fetch that link. Check the URL is correct and reachable.",
        ) from exc

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in ctype:
        resp.close()
        raise HTTPException(
            status_code=400,
            detail="That link returned a web page, not CSV. In Google Sheets use "
                   "File → Share → Publish to web → CSV, then paste that link.",
        )

    tmp = tempfile.NamedTemporaryFile(prefix="datapulse_url_", suffix=".csv", delete=False)
    size = 0
    try:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"The linked file is too large. The limit is "
                           f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            tmp.write(chunk)
        tmp.close()
        if size == 0:
            raise HTTPException(status_code=400, detail="The linked file is empty.")
        return tmp.name
    except BaseException:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    finally:
        resp.close()


def _name_from_url(url: str) -> str:
    """A friendly default dataset name derived from the URL, else a generic label."""
    try:
        base = os.path.basename(urlparse(url).path)
        if base and "." in base:
            return base
    except Exception:  # noqa: BLE001
        pass
    return "Linked CSV"


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


class UrlRequest(BaseModel):
    url: str
    name: Optional[str] = None


@app.post("/datasets/url")
def create_url_dataset(req: UrlRequest) -> dict[str, Any]:
    """Load a public CSV URL (e.g. a Google Sheet published to the web as CSV).

    The URL is fetched server-side through the same parsing/schema-detection path
    as an upload, and stored on the dataset so it can be re-fetched via /refresh.
    """
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Please provide a CSV link.")
    url = req.url.strip()
    path = _fetch_csv_url(url)
    try:
        ds = _create_dataset(path, name=(req.name or _name_from_url(url)),
                             source="url", source_url=url)
        return ds.public()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/datasets/{dataset_id}/refresh")
def refresh_dataset(dataset_id: str) -> dict[str, Any]:
    """Re-fetch a URL-backed dataset's source and replace its data in place.

    Keeps the same dataset id (so the open view/filters stay valid). The new data
    is loaded into a temp table first; the live table is only swapped out once the
    fetch and parse succeed, so a failed refresh leaves the existing data intact.
    """
    ds = _get_dataset(dataset_id)
    if ds.source != "url" or not ds.source_url:
        raise HTTPException(
            status_code=400,
            detail="This dataset has no source link to refresh. Refresh works on datasets "
                   "loaded from a CSV link.",
        )
    path = _fetch_csv_url(ds.source_url)
    tmp_table = f"ds_{uuid.uuid4().hex}"
    try:
        assert con is not None
        with _lock:
            columns, row_count = _build_table_locked(tmp_table, path, None)
            con.execute(f"DROP TABLE IF EXISTS {_qi(ds.table)}")
            con.execute(f"ALTER TABLE {_qi(tmp_table)} RENAME TO {_qi(ds.table)}")
            ds.columns = columns
            ds.row_count = row_count
            ds.last_access = time.time()
        return ds.public()
    finally:
        try:
            os.unlink(path)
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
    # Expose DuckDB's stable rowid as __rowid so Evidence Mode can highlight the
    # exact rows an insight is built on (it never renders as a normal column).
    cur.execute(
        f"SELECT *, rowid AS __rowid FROM {_qi(ds.table)}{where}{order} LIMIT ? OFFSET ?",
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
    split_by: Optional[str] = Query(None, description="Category column: one time-series line per value."),
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

        agg_expr = "COUNT(*)" if agg == "count" else f"{agg.upper()}({_qi(y_column)})"

        # Optional "split by": draw one line per value of a category column. We
        # cap the number of lines to the top-N categories (by row count) so the
        # chart stays fast and readable regardless of cardinality.
        if split_by:
            if split_by not in ds.column_names:
                raise HTTPException(status_code=400, detail=f"Unknown split_by column {split_by!r}.")
            if split_by == column:
                raise HTTPException(status_code=400, detail="split_by must differ from the date column.")
            scol = _qi(split_by)
            key_expr = f"COALESCE(CAST({scol} AS VARCHAR), '(blank)')"
            top = cur.execute(
                f"SELECT {key_expr} AS k, COUNT(*) c FROM {tbl}{where} "
                f"GROUP BY k ORDER BY c DESC LIMIT {SPLIT_SERIES_TOP_N}", params
            ).fetchall()
            keys = [k for k, _ in top]
            if not keys:
                return {"chart_type": chart_type, "column": column, "interval": chosen,
                        "agg": agg, "y_column": y_column, "split_by": split_by,
                        "labels": [], "series": []}
            placeholders = ", ".join("?" for _ in keys)
            sql = (
                f"SELECT date_trunc('{chosen}', {col}) AS bucket, {key_expr} AS k, {agg_expr} AS value "
                f"FROM {tbl}{_augment_where(where, f'{key_expr} IN ({placeholders})')} "
                f"GROUP BY bucket, k ORDER BY bucket"
            )
            rows = cur.execute(sql, [*params, *keys]).fetchall()
            buckets: list[Any] = []
            seen: set = set()
            grid: dict[Any, dict[Any, float]] = {k: {} for k in keys}
            for bucket, k, v in rows:
                if bucket not in seen:
                    seen.add(bucket)
                    buckets.append(bucket)
                grid.setdefault(k, {})[bucket] = round(float(v), 4) if v is not None else None
            labels = [_jsonable(b) for b in buckets]
            series = [
                {"key": k, "values": [grid.get(k, {}).get(b) for b in buckets]}
                for k in keys
            ]
            return {"chart_type": chart_type, "column": column, "interval": chosen,
                    "agg": agg, "y_column": y_column, "split_by": split_by,
                    "labels": labels, "series": series}

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


def _compare_metric_expr(ds: Dataset, agg: str, metric_column: Optional[str]) -> str:
    """Validated aggregate expression used by Compare Mode."""
    if agg == "count":
        return "COUNT(*)"
    if agg not in {"sum", "avg"}:
        raise HTTPException(status_code=400, detail="agg must be count, sum or avg.")
    if not metric_column or metric_column not in ds.column_names:
        raise HTTPException(status_code=400, detail="A numeric metric_column is required for sum/avg.")
    if ds.type_of(metric_column) != "number":
        raise HTTPException(status_code=400, detail=f"{metric_column!r} is not a numeric column.")
    return f"{agg.upper()}({_qi(metric_column)})"


def _metric_value(value: Any, agg: str) -> float | int:
    if value is None:
        return 0
    if agg == "count":
        return int(value)
    return round(float(value), 4)


@app.get("/datasets/{dataset_id}/compare")
def dataset_compare(
    dataset_id: str,
    agg: str = Query("count", description="count | sum | avg"),
    metric_column: Optional[str] = None,
    date_column: Optional[str] = None,
    dimension_column: Optional[str] = None,
    interval: Optional[str] = None,
    filters: Optional[str] = None,
) -> dict[str, Any]:
    """Compact aggregate snapshot for Compare Mode.

    The frontend calls this twice (baseline/current) and computes deltas locally.
    Keeping the heavy aggregation in DuckDB avoids shipping large row payloads to
    the browser while preserving the existing upload/table/chart endpoints.
    """
    ds = _get_dataset(dataset_id)
    cur = _cursor()
    metric_expr = _compare_metric_expr(ds, agg, metric_column)
    where, params = _build_where(ds, filters)
    tbl = _qi(ds.table)

    rows, total = cur.execute(
        f"SELECT COUNT(*), {metric_expr} FROM {tbl}{where}", params
    ).fetchone()

    series: list[dict[str, Any]] = []
    chosen_interval: Optional[str] = None
    if date_column:
        if date_column not in ds.column_names:
            raise HTTPException(status_code=400, detail=f"Unknown date column {date_column!r}.")
        if ds.type_of(date_column) != "date":
            raise HTTPException(status_code=400, detail=f"{date_column!r} is not a date column.")
        date_ident = _qi(date_column)
        min_ts, max_ts = cur.execute(
            f"SELECT MIN({date_ident}), MAX({date_ident}) FROM {tbl}{where}", params
        ).fetchone()
        if min_ts is not None and max_ts is not None:
            span = max((max_ts - min_ts).total_seconds(), 1.0)
            chosen_interval = _pick_interval(span, interval)
            sql = (
                f"SELECT date_trunc('{chosen_interval}', {date_ident}) AS bucket, "
                f"{metric_expr} AS value FROM {tbl}{where} GROUP BY bucket ORDER BY bucket"
            )
            series = [
                {"label": _jsonable(bucket), "value": _metric_value(value, agg)}
                for bucket, value in cur.execute(sql, params).fetchall()
            ]

    drivers: list[dict[str, Any]] = []
    if dimension_column:
        if dimension_column not in ds.column_names:
            raise HTTPException(status_code=400, detail=f"Unknown dimension column {dimension_column!r}.")
        dim = _qi(dimension_column)
        sql = (
            f"SELECT CAST({dim} AS VARCHAR) AS label, {metric_expr} AS metric_value "
            f"FROM {tbl}{where} GROUP BY label ORDER BY ABS(metric_value) DESC LIMIT 30"
        )
        drivers = [
            {"label": str(label) if label is not None else "Blank", "value": _metric_value(value, agg)}
            for label, value in cur.execute(sql, params).fetchall()
        ]

    return {
        "agg": agg,
        "metric_column": metric_column,
        "date_column": date_column,
        "dimension_column": dimension_column,
        "row_count": int(rows),
        "total": _metric_value(total, agg),
        "interval": chosen_interval,
        "series": series,
        "drivers": drivers,
    }


# ---------------------------------------------------------------------------
# Evidence Mode — evidence-backed insights.
# ---------------------------------------------------------------------------

@app.get("/datasets/{dataset_id}/insights")
def dataset_insights(
    dataset_id: str,
    mode: str = Query("analyst", description="student | analyst | founder | manager | researcher"),
    filters: Optional[str] = None,
) -> dict[str, Any]:
    """Compute the Evidence-Mode report: every insight is backed by real math,
    exact numbers and the actual row ids that prove it (see insights.py).

    compute_report is self-guarding (per-section try/except + time budget), but we
    also wrap the call here so a truly unexpected failure still returns HTTP 200
    with a valid, honest report instead of a 500/503 that blanks the Evidence tab.
    """
    ds = _get_dataset(dataset_id)
    where, params = _build_where(ds, filters)
    if con is None:
        raise HTTPException(status_code=503, detail="Server not ready.")
    try:
        report = insights_mod.compute_report(con, ds.table, ds.columns, where, params, mode)
    except Exception:  # noqa: BLE001 - defence in depth; compute_report shouldn't raise
        import traceback
        traceback.print_exc()
        report = {
            "summary": {"rows": ds.row_count, "columns": len(ds.columns),
                        "column_types": {c["name"]: c["type"] for c in ds.columns},
                        "date_range": None, "key_numbers": {}},
            "data_quality": {"missing_by_column": {}, "duplicate_rows": 0,
                             "duplicate_example_rows": [],
                             "small_sample_warning": {"triggered": False, "row_count": ds.row_count,
                                                      "threshold": 30,
                                                      "message": "Insights could not be computed for this dataset."}},
            "insights": [{
                "id": "insights_unavailable", "title": "Insights unavailable",
                "category": "executive_summary",
                "explanation": "The insight engine hit an unexpected error on this dataset. "
                               "The rest of DataPulse (table, charts, export) still works.",
                "why_it_matters": "No claim is shown because the computation did not complete.",
                "confidence": 0.0, "trust_score": 0, "evidence_rows": [],
                "evidence_columns": [c["name"] for c in ds.columns], "supporting_metrics": {},
                "what_to_check_next": "Try a different slice, or re-upload the file.",
                "is_limitation": True,
            }],
            "follow_up_questions": ["Is the data well-formed enough to analyse?"],
        }
    report["dataset_id"] = ds.id
    report["mode"] = mode if mode in insights_mod._MODES else "analyst"
    return report


@app.get("/datasets/{dataset_id}/rows")
def dataset_rows(
    dataset_id: str,
    rowids: str = Query(..., description="Comma-separated DuckDB rowids to fetch"),
    filters: Optional[str] = None,
) -> dict[str, Any]:
    """Return the full data for specific rowids so the evidence drawer can show
    the exact rows behind a claim. Ids are bound, never interpolated."""
    ds = _get_dataset(dataset_id)
    cur = _cursor()
    try:
        ids = [int(x) for x in rowids.split(",") if x.strip() != ""][:200]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="rowids must be comma-separated integers.") from exc
    if not ids:
        return {"data": [], "columns": ds.columns}

    where, params = _build_where(ds, filters)
    placeholders = ",".join("?" for _ in ids)
    clause = f"rowid IN ({placeholders})"
    where = f"{where} AND {clause}" if where else f" WHERE {clause}"
    cur.execute(
        f"SELECT *, rowid AS __rowid FROM {_qi(ds.table)}{where} ORDER BY rowid",
        [*params, *ids],
    )
    cols = [d[0] for d in cur.description]
    rows = [{k: _jsonable(v) for k, v in zip(cols, r)} for r in cur.fetchall()]
    # Preserve the caller's requested order (rowid order from SQL may differ).
    by_id = {r["__rowid"]: r for r in rows}
    ordered = [by_id[i] for i in ids if i in by_id]
    return {"data": ordered, "columns": ds.columns}


# ---------------------------------------------------------------------------
# Shared reports — persist a computed report and serve it read-only by token.
# ---------------------------------------------------------------------------

class ReportSubmission(BaseModel):
    report: dict[str, Any]
    dataset_name: Optional[str] = None


_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")


@app.post("/reports")
def create_report(body: ReportSubmission) -> dict[str, Any]:
    """Store a computed Evidence-Mode report under a uuid token and return a
    read-only URL. Persistence is best-effort (see REPORTS_DIR note)."""
    payload = {
        "report": body.report,
        "dataset_name": body.dataset_name,
        "created_at": time.time(),
    }
    encoded = json.dumps(payload).encode("utf-8")
    if len(encoded) > MAX_REPORT_BYTES:
        raise HTTPException(status_code=413, detail="Report is too large to share.")

    token = uuid.uuid4().hex
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(os.path.join(REPORTS_DIR, f"{token}.json"), "wb") as fh:
            fh.write(encoded)
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not save the report on this server (storage is ephemeral "
                   "on the free tier). Please try again.",
        ) from exc
    return {"token": token, "url": f"/report/{token}"}


@app.get("/reports/{token}")
def get_report(token: str) -> dict[str, Any]:
    """Return a previously stored report by token (read-only)."""
    if not _TOKEN_RE.match(token):  # tokens are uuid4 hex -> no path traversal
        raise HTTPException(status_code=400, detail="Invalid report token.")
    path = os.path.join(REPORTS_DIR, f"{token}.json")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="This shared report was not found. Links are best-effort and may "
                   "expire when the free-tier server restarts.",
        )
    with open(path, "rb") as fh:
        return json.loads(fh.read())


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the DataPulse API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
