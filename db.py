"""Persistent metadata store for datasets and saved reports.

Uses the Supabase Postgres given by ``DATABASE_URL`` in production; falls back to
a local SQLite file when ``DATABASE_URL`` is unset or ``sqlite:///…`` (handy for
local dev and CI). Only *metadata* lives here — the analysis itself still runs on
in-memory DuckDB. Persisting the schema and connected source URL lets URL-backed
datasets survive restarts and be reloaded/refreshed per user.

Every row carries a ``user_id`` so data is strictly private per user; callers pass
the authenticated user id and the API enforces ownership.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Optional

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

_lock = threading.RLock()
_conn = None
_backend: Optional[str] = None  # "postgres" | "sqlite"


def _is_postgres(url: str) -> bool:
    return url.startswith(("postgres://", "postgresql://", "postgresql+"))


def _q(sql: str) -> str:
    """Translate ``?`` placeholders to ``%s`` for psycopg (Postgres)."""
    return sql.replace("?", "%s") if _backend == "postgres" else sql


def init_db() -> None:
    """Open the connection (idempotent) and create tables if needed."""
    global _conn, _backend
    with _lock:
        if _conn is not None:
            return
        if DATABASE_URL and _is_postgres(DATABASE_URL):
            import psycopg  # lazy: only needed when a Postgres URL is configured

            dsn = DATABASE_URL
            for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://", "postgres://"):
                if dsn.startswith(prefix):
                    dsn = "postgresql://" + dsn.split("://", 1)[1]
                    break
            _conn = psycopg.connect(dsn, autocommit=True)
            _backend = "postgres"
        else:
            path = DATABASE_URL
            if path.startswith("sqlite:///"):
                path = path[len("sqlite:///"):]
            path = path or "./datapulse.db"
            _conn = sqlite3.connect(path, check_same_thread=False)
            _conn.isolation_level = None  # autocommit
            _backend = "sqlite"
        _create_tables()


def _exec(sql: str, params: tuple = ()):  # noqa: ANN001
    with _lock:
        return _conn.execute(_q(sql), params)


def _create_tables() -> None:
    _exec(
        """CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT,
            source TEXT,
            source_url TEXT,
            schema_json TEXT,
            row_count INTEGER,
            created_at DOUBLE PRECISION
        )"""
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_datasets_user ON datasets (user_id)")
    _exec(
        """CREATE TABLE IF NOT EXISTS reports (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            payload_json TEXT,
            created_at DOUBLE PRECISION
        )"""
    )
    _exec("CREATE INDEX IF NOT EXISTS idx_reports_user ON reports (user_id)")


# --- datasets --------------------------------------------------------------

def save_dataset(id: str, user_id: str, name: str, source: str,
                 source_url: Optional[str], schema: list, row_count: int,
                 created_at: Optional[float] = None) -> None:
    """Insert or update a dataset's metadata (upsert on id)."""
    created_at = created_at if created_at is not None else time.time()
    schema_json = json.dumps(schema)
    with _lock:
        _exec("DELETE FROM datasets WHERE id = ?", (id,))
        _exec(
            "INSERT INTO datasets (id, user_id, name, source, source_url, schema_json, "
            "row_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (id, user_id, name, source, source_url, schema_json, int(row_count), created_at),
        )


def _row_to_dataset(row) -> dict:
    return {
        "id": row[0], "user_id": row[1], "name": row[2], "source": row[3],
        "source_url": row[4], "schema": json.loads(row[5]) if row[5] else [],
        "row_count": row[6], "created_at": row[7],
    }


def get_dataset(id: str) -> Optional[dict]:
    cur = _exec(
        "SELECT id, user_id, name, source, source_url, schema_json, row_count, created_at "
        "FROM datasets WHERE id = ?", (id,)
    )
    row = cur.fetchone()
    return _row_to_dataset(row) if row else None


def list_datasets(user_id: str) -> list[dict]:
    cur = _exec(
        "SELECT id, user_id, name, source, source_url, schema_json, row_count, created_at "
        "FROM datasets WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
    )
    return [_row_to_dataset(r) for r in cur.fetchall()]


def delete_dataset(id: str) -> None:
    _exec("DELETE FROM datasets WHERE id = ?", (id,))


# --- reports ---------------------------------------------------------------

def save_report(token: str, user_id: str, payload: dict) -> None:
    _exec(
        "INSERT INTO reports (token, user_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
        (token, user_id, json.dumps(payload), time.time()),
    )


def get_report(token: str) -> Optional[dict]:
    cur = _exec("SELECT token, user_id, payload_json FROM reports WHERE token = ?", (token,))
    row = cur.fetchone()
    if not row:
        return None
    return {"token": row[0], "user_id": row[1], "payload": json.loads(row[2])}


# --- account deletion (GDPR) -----------------------------------------------

def delete_user_data(user_id: str) -> list[str]:
    """Delete every dataset and report owned by ``user_id``.

    Returns the ids of the deleted datasets so the caller can also drop any live
    in-memory tables. Scoped strictly to the given user — nothing else is touched.
    """
    with _lock:
        cur = _exec("SELECT id FROM datasets WHERE user_id = ?", (user_id,))
        dataset_ids = [r[0] for r in cur.fetchall()]
        _exec("DELETE FROM datasets WHERE user_id = ?", (user_id,))
        _exec("DELETE FROM reports WHERE user_id = ?", (user_id,))
    return dataset_ids
