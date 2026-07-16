"""Evidence Mode — evidence-backed insight computation for DataPulse.

Every insight this module emits is produced by real computation over the
dataset (DuckDB SQL + a little numpy), never by an LLM. Each insight carries the
exact numbers (`supporting_metrics`), the columns involved (`evidence_columns`)
and the actual base-table row ids (`evidence_rows`) that prove the claim, so the
frontend can show the proof and highlight those rows in the table.

If the data is too small, too messy, or a check does not apply, we emit a
clearly-labelled limitation instead of a claim (``is_limitation: True``).

Design notes
------------
* We reference rows by DuckDB's stable per-table ``rowid`` (0-based). The
  ``/query`` and ``/rows`` endpoints expose the same rowid as ``__rowid`` so the
  UI can match evidence to visible rows.
* The audience ``mode`` only changes the wording/emphasis of ``explanation`` and
  ``why_it_matters`` — never the underlying math. All numbers are computed once.

Confidence & trust — the two scores, defined once and reused everywhere:

    confidence (0..1) = sample_weight * effect_size
        sample_weight = min(1, n / 50)      # ~50 usable rows -> full weight
        effect_size   = standardized strength of THIS finding, in [0, 1]
                        (e.g. |r| for a correlation, the dominant share for a
                        concentration, R^2 for a trend, outlier severity, ...)
      -> a finding is only confident when we have enough rows AND a real effect.

    trust_score (0..100) = 100 * sample_weight * completeness_weight * consistency_weight
        sample_weight       = min(1, n / 50)                 # more rows -> more trust
        completeness_weight = 1 - avg_missing_fraction(cols)  # fewer nulls -> more trust
        consistency_weight  = stability of the effect, in [0, 1]
                              (e.g. |r|, R^2, 1 - outlier_fraction, share)
      -> trust rewards big, clean, consistent evidence and punishes tiny or
         null-riddled slices even when an effect looks large.

Low confidence is always stated plainly in the ``explanation`` text.
"""

from __future__ import annotations

import concurrent.futures as _futures
import logging
import os
import re
from datetime import date, datetime, timezone
from decimal import Decimal
from itertools import combinations
from typing import Any, Optional

log = logging.getLogger("datapulse.insights")

# Hard per-section time budget. Each insight section runs on its own cursor in a
# worker thread; if it exceeds this it is abandoned and shown as a limitation, so
# one slow/pathological query can never hang the request (see _guarded).
SECTION_BUDGET_S = float(os.getenv("DATAPULSE_INSIGHT_SECTION_BUDGET", "12"))
_POOL = _futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="insight")

# ---------------------------------------------------------------------------
# Thresholds (all tunable in one place).
# ---------------------------------------------------------------------------
SMALL_SAMPLE = 30          # below this we warn that findings are unreliable
MISSING_FLAG = 0.20        # a column missing >20% of values is flagged
CONCENTRATION_FLAG = 0.40  # a category holding >40% of rows is "dominant"
MIN_OUTLIER_N = 20         # need at least this many rows for IQR outlier logic
MIN_CORR_N = 10            # need at least this many paired rows for a correlation
STRONG_CORR = 0.30         # |r| at/above this is reported as a real relationship
MIN_HALF_N = 5             # min rows per half for what-changed-most
EVIDENCE_CAP = 25          # max row ids attached to a single insight
FULL_SAMPLE = 50.0         # rows at which sample_weight saturates to 1.0

# Correlation title bands by |r| (see _corr_relation).
WEAK_CORR = 0.20           # below this two columns are reported as "unrelated"
MODERATE_CORR = 0.40       # 0.40..0.69 is "moderately related"
STRONG_CORR_TITLE = 0.70   # at/above this is "strongly related"

# Top-insights gating (Fix 2): drop nothing-burgers, don't pad a weak headline.
TOP_MIN_TRUST = 20         # a headline "Top insight" must beat this trust score
TOP_STRONG_TRUST = 40      # need >=2 findings above this to justify a Top section

# Identifier detection (Fix 4).
ID_UNIQUE_FRAC = 0.95      # distinct >= 95% of rows -> treated as an identifier
_ID_NAME_TOKENS = {"id", "uuid", "key", "index"}


# ---------------------------------------------------------------------------
# Small helpers (kept local so this module has no import cycle with main).
# ---------------------------------------------------------------------------

def _qi(identifier: str) -> str:
    """Quote a SQL identifier, escaping embedded double quotes."""
    return '"' + identifier.replace('"', '""') + '"'


def _j(v: Any) -> Any:
    """JSON-safe scalar (datetime -> iso, Decimal -> float)."""
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _augment(where: str, extra: Optional[str]) -> str:
    """Append a server-built predicate to an existing WHERE clause."""
    if not extra:
        return where
    return f"{where} AND {extra}" if where.strip() else f" WHERE {extra}"


def _round(x: Any, n: int = 4) -> Any:
    try:
        if x is None:
            return None
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def _sample_weight(n: int) -> float:
    return min(1.0, n / FULL_SAMPLE)


def _confidence(n: int, effect: float) -> float:
    """See module docstring. effect is the standardized strength in [0, 1]."""
    return round(max(0.0, min(1.0, _sample_weight(n) * max(0.0, min(1.0, effect)))), 3)


def _trust(n: int, missing_frac: float, consistency: float) -> int:
    """See module docstring. Returns an int 0..100."""
    sample_w = _sample_weight(n)
    completeness_w = max(0.0, 1.0 - max(0.0, missing_frac))
    consistency_w = max(0.0, min(1.0, consistency))
    return int(round(100 * sample_w * completeness_w * consistency_w))


def _corr_relation(r: float) -> tuple[str, str]:
    """Map a Pearson r to an honest (relation_word, strength_word) pair.

    The relation word drives the TITLE and must reflect |r|, never overclaim:
    <0.20 unrelated · 0.20–0.39 weakly · 0.40–0.69 moderately · >=0.70 strongly.
    """
    ar = abs(r)
    if ar < WEAK_CORR:
        return "unrelated", "negligible"
    if ar < MODERATE_CORR:
        return "weakly related", "weak"
    if ar < STRONG_CORR_TITLE:
        return "moderately related", "moderate"
    return "strongly related", "strong"


def _looks_like_id_name(name: str) -> bool:
    """True if a column name reads like an identifier (id, _id, uuid, key, index)."""
    low = name.lower()
    if low == "id" or low.endswith("_id"):
        return True
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    return any(t in _ID_NAME_TOKENS for t in tokens)


def _identifier_columns(cur, table, numeric, integer_cols, where, params, total) -> set[str]:
    """Numeric columns that are identifiers, not measurements — excluded from
    correlation and outlier analysis (Fix 4).

    A column is flagged when it (a) is named like an id, (b) has (near-)unique
    values (distinct >= 95% of rows), or (c) is strictly monotonic increasing by
    rowid. The cardinality/order tests (b, c) only apply to INTEGER-typed columns:
    real ids, keys, indexes and sequences are integers (or text), whereas a
    continuous float measurement (revenue, price, latency) is routinely
    all-distinct and must NOT be mistaken for an identifier. Best-effort: a probe
    failing on one column just leaves it unflagged.
    """
    ids: set[str] = set()
    for name in numeric:
        if _looks_like_id_name(name):
            ids.add(name)
            continue
        if name not in integer_cols:
            continue  # float/decimal measurement — never an id by shape alone
        ident = _qi(name)
        try:
            nd, nn = cur.execute(
                f"SELECT COUNT(DISTINCT {ident}), COUNT({ident}) FROM {_qi(table)}{where}", params
            ).fetchone()
            nd, nn = int(nd or 0), int(nn or 0)
            if total and nd >= ID_UNIQUE_FRAC * total:
                ids.add(name)
                continue
            # strictly increasing by rowid -> a sequence/index, not a measurement
            if nn >= 2:
                w = _augment(where, f"{ident} IS NOT NULL")
                viol = int(cur.execute(
                    f"SELECT COUNT(*) FROM (SELECT {ident} AS v, "
                    f"LAG({ident}) OVER (ORDER BY rowid) AS p FROM {_qi(table)}{w}) "
                    f"WHERE p IS NOT NULL AND v <= p", params
                ).fetchone()[0])
                if viol == 0:
                    ids.add(name)
        except Exception as exc:  # noqa: BLE001 - a bad probe just means "not flagged"
            log.warning("insights: identifier probe failed for %r: %r", name, exc)
    return ids


# ---------------------------------------------------------------------------
# Audience wording. Changes ONLY tone/emphasis, never the numbers.
# ---------------------------------------------------------------------------
# Each entry is a short lens applied to why_it_matters; the factual explanation
# (with the numbers) is shared across modes and only gets a light lead-in.
_MODES = ("student", "analyst", "founder", "manager", "researcher")

_LEAD_IN = {
    "student": "Here's what this means in plain terms: ",
    "analyst": "",
    "founder": "",
    "manager": "",
    "researcher": "",
}

# The `explanation` always leads with plain meaning. For these two audiences we
# append ONE short technical clause carrying the raw stat; everyone else sees the
# plain sentence only. The full numbers always live in supporting_metrics too.
_TECHNICAL_MODES = ("analyst", "researcher")


def _tech(mode: str, clause: str) -> str:
    """Return a short technical clause for analyst/researcher, '' for everyone else."""
    return clause if mode in _TECHNICAL_MODES else ""


def _why(category: str, mode: str, f: dict[str, Any]) -> str:
    """Audience-specific 'why it matters', formatted from computed facts ``f``."""
    templates = _WHY_TEMPLATES.get(category, {})
    text = templates.get(mode) or templates.get("analyst") or ""
    try:
        return text.format(**f)
    except (KeyError, IndexError):
        return text


# Templates are intentionally verbose per audience but read from the SAME facts.
_WHY_TEMPLATES: dict[str, dict[str, str]] = {
    "executive_summary": {
        "student": "It tells you how big the dataset is and what kind of columns you're working with before you dig in.",
        "analyst": "Knowing the shape, types and date span up front prevents mis-scoped analysis later.",
        "founder": "This is the size and coverage of the data behind any decision you make from it.",
        "manager": "A quick read on scope and coverage so you know how much weight this dataset can bear.",
        "researcher": "Establishes n, variable types and the observation window that bound every downstream claim.",
    },
    "missing_data": {
        "student": "Missing values can quietly skew averages and charts, so it's good to know where the gaps are.",
        "analyst": "Columns with heavy nulls bias aggregates and correlations; treat their metrics with caution.",
        "founder": "Gaps here mean any metric built on these fields is only partly backed by real data.",
        "manager": "Decisions leaning on these columns rest on incomplete data — worth flagging to the team.",
        "researcher": "Missingness threatens validity; the null rate below indicates where imputation or exclusion is needed.",
    },
    "correlation": {
        "student": "When two numbers move together, one might help predict the other — but it isn't proof one causes the other.",
        "analyst": "A strong pairwise relationship is a modelling lead and a redundancy check, not evidence of causation.",
        "founder": "These two metrics tend to move together, which can hint at a lever worth testing.",
        "manager": "A relationship worth probing before you act on either metric in isolation.",
        "researcher": "Report r with n; association is not causation and may reflect a confound.",
    },
    "anomalies": {
        "student": "A few unusual rows can pull the average around; it's worth checking if they're real or typos.",
        "analyst": "Outliers distort means and variance — decide to keep, cap or investigate before aggregating.",
        "founder": "These extreme rows could be your best customers, fraud, or data errors — each needs a different response.",
        "manager": "Exceptions here can dominate totals; confirm whether they're signal or noise before reporting.",
        "researcher": "Flagged points exceed the IQR fence; assess whether they are errors or genuine tail observations.",
    },
    "concentration": {
        "student": "One group makes up a big slice of the data, so overall numbers mostly reflect that group.",
        "analyst": "High concentration means aggregate metrics are effectively driven by one segment.",
        "founder": "A big share of everything comes from one group — concentration risk and opportunity in one.",
        "manager": "The headline numbers largely describe this dominant group, not the long tail.",
        "researcher": "Class imbalance of this magnitude will bias any model or mean toward the majority group.",
    },
    "trend": {
        "student": "The values are generally going up or down over time, not staying flat.",
        "analyst": "A directional trend over the date column shapes forecasting and seasonality checks.",
        "founder": "The direction of travel over time — the momentum behind this metric.",
        "manager": "Whether things are improving or sliding over the period covered.",
        "researcher": "Linear trend estimate over time; inspect residuals and R^2 before extrapolating.",
    },
    "time_pattern": {
        "student": "Knowing which day is busiest helps you plan around the rush.",
        "analyst": "A weekly cycle this strong should be modelled (or removed) before reading any trend.",
        "founder": "Your activity has a rhythm — the day it peaks is where staffing and pushes pay off.",
        "manager": "The workload isn't even across the week; resourcing should follow the peak.",
        "researcher": "A day-of-week effect of this size confounds level comparisons; control for it.",
    },
    "what_changed": {
        "student": "Comparing the earlier half to the later half shows which group changed the most.",
        "analyst": "First-vs-second-half deltas isolate the segments driving the overall movement.",
        "founder": "What moved the needle between the earlier and later period — where growth or decline came from.",
        "manager": "The segments responsible for most of the change across the period.",
        "researcher": "Period-split contrast identifies the largest between-period shifts by group.",
    },
    "data_quality": {
        "student": "Duplicate or missing rows can make things look bigger or cleaner than they are.",
        "analyst": "Duplicates inflate counts and missingness biases aggregates — quantify both before trusting totals.",
        "founder": "Rough edges in the data that could distort any number you quote from it.",
        "manager": "Data hygiene issues that could undermine confidence in the figures.",
        "researcher": "Duplication and missingness are threats to data integrity that must be reported with n.",
    },
}


# ---------------------------------------------------------------------------
# Row-evidence helpers.
# ---------------------------------------------------------------------------

def _rowids(cur, table, base_where, base_params, extra_sql=None,
            extra_params=(), order_by="rowid", limit=EVIDENCE_CAP) -> list[int]:
    """Fetch up to ``limit`` base-table rowids matching where + an extra predicate."""
    where = _augment(base_where, extra_sql)
    params = [*base_params, *extra_params]
    sql = f"SELECT rowid FROM {_qi(table)}{where} ORDER BY {order_by} LIMIT {int(limit)}"
    return [int(r[0]) for r in cur.execute(sql, params).fetchall()]


def _insight(**kw) -> dict[str, Any]:
    """Build an insight dict with all required keys defaulted (keeps shape stable)."""
    base = {
        "id": kw.get("id", ""),
        "title": kw.get("title", ""),
        "category": kw.get("category", ""),
        "explanation": kw.get("explanation", ""),
        "why_it_matters": kw.get("why_it_matters", ""),
        "confidence": kw.get("confidence", 0.0),
        "trust_score": kw.get("trust_score", 0),
        "evidence_rows": kw.get("evidence_rows", []),
        "evidence_columns": kw.get("evidence_columns", []),
        "supporting_metrics": kw.get("supporting_metrics", {}),
        "what_to_check_next": kw.get("what_to_check_next", ""),
        "is_limitation": kw.get("is_limitation", False),
    }
    # internal-only ranking signal, stripped before returning to the client
    base["_notability"] = kw.get("notability", base["confidence"] * base["trust_score"] / 100.0)
    return base


def _limitation(name: str, category: str, cols_hint, reason: str) -> dict:
    """A clearly-labelled placeholder card emitted when a section can't be computed.

    Keeps the report honest and complete instead of failing the whole request:
    the section is shown as a limitation (no claim, no numbers) rather than dropped.
    """
    return _insight(
        id=f"{category}_unavailable",
        title=f"{name} unavailable",
        category=category,
        explanation=f"The {name.lower()} check {reason}. It is shown as a limitation "
                    "rather than a claim, so nothing here is asserted without evidence.",
        why_it_matters="No conclusion is drawn from this section because its computation "
                       "did not complete.",
        confidence=0.0, trust_score=0, is_limitation=True,
        evidence_columns=list(cols_hint or []),
        what_to_check_next="Retry, or try a cleaner or larger slice of the data.",
        notability=0.0,
    )


def _guarded(con, name: str, category: str, cols_hint, fn) -> list[dict]:
    """Run one insight section on its own cursor under a hard time budget.

    If it raises OR exceeds SECTION_BUDGET_S, we log it and return a limitation
    card for that section instead of letting the exception kill the request. This
    is what guarantees a single broken/slow check can never 500/503 the endpoint.
    """
    def work():
        return fn(con.cursor())
    try:
        return _POOL.submit(work).result(timeout=SECTION_BUDGET_S) or []
    except _futures.TimeoutError:
        log.warning("insights: section %r exceeded %.0fs budget — degraded to limitation",
                    name, SECTION_BUDGET_S)
        return [_limitation(name, category, cols_hint,
                            f"took longer than {int(SECTION_BUDGET_S)}s and was skipped to keep the report responsive")]
    except Exception as exc:  # noqa: BLE001 - any failure degrades, never propagates
        log.warning("insights: section %r failed: %r", name, exc, exc_info=True)
        return [_limitation(name, category, cols_hint, f"could not be computed ({type(exc).__name__})")]


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def compute_report(con, table: str, columns: list[dict], where: str = "",
                   params: Optional[list] = None, mode: str = "analyst") -> dict[str, Any]:
    """Compute the full Evidence-Mode report for one (optionally filtered) dataset.

    ``con`` is a DuckDB connection (each section gets its own cursor from it),
    ``table`` the internal table name, ``columns`` the dataset's
    ``[{name, type, sql_type}]`` list, and ``where``/``params`` come from the shared
    filter builder.

    Robustness contract: this ALWAYS returns a valid, JSON-serialisable report.
    Every section is isolated (try/except + time budget); a catastrophic failure
    still yields a minimal report rather than raising.
    """
    params = list(params or [])
    if mode not in _MODES:
        mode = "analyst"

    numeric = [c["name"] for c in columns if c["type"] == "number"]
    dates = [c["name"] for c in columns if c["type"] == "date"]
    texts = [c["name"] for c in columns if c["type"] == "text"]
    all_names = [c["name"] for c in columns]

    try:
        return _finalize(_compute(con, table, columns, where, params, mode,
                                  numeric, dates, texts, all_names))
    except Exception as exc:  # noqa: BLE001 - last-resort guard: never raise to the caller
        log.error("insights: catastrophic failure, returning minimal report: %r", exc, exc_info=True)
        return _finalize({
            "summary": {"rows": None, "columns": len(all_names),
                        "column_types": {c["name"]: c["type"] for c in columns},
                        "date_range": None, "key_numbers": {}},
            "data_quality": {"missing_by_column": {}, "duplicate_rows": 0,
                             "duplicate_example_rows": [],
                             "small_sample_warning": {"triggered": False, "row_count": None,
                                                      "threshold": SMALL_SAMPLE,
                                                      "message": "Data quality could not be assessed."}},
            "insights": [_limitation("Insights", "executive_summary", all_names,
                                     "could not be computed for this dataset")],
            "follow_up_questions": ["Is the data well-formed enough to analyse?"],
        })


def _compute(con, table, columns, where, params, mode, numeric, dates, texts, all_names) -> dict:
    """Inner body of compute_report; wrapped by the last-resort guard above."""
    base = con.cursor()
    total = int(base.execute(f"SELECT COUNT(*) FROM {_qi(table)}{where}", params).fetchone()[0])

    # ---- data quality: per-column missingness + duplicate rows (each guarded) --
    try:
        missing_by_column = _missing_by_column(con.cursor(), table, columns, where, params, total)
    except Exception as exc:  # noqa: BLE001
        log.warning("insights: missing-by-column failed: %r", exc)
        missing_by_column = {c["name"]: {"missing": 0, "missing_fraction": 0.0,
                                         "missing_pct": 0.0, "type": c["type"]} for c in columns}
    missing_frac = {k: v["missing_fraction"] for k, v in missing_by_column.items()}
    try:
        duplicate_rows, dup_example_rows = _duplicates(con.cursor(), table, all_names, where, params, total)
    except Exception as exc:  # noqa: BLE001
        log.warning("insights: duplicate detection failed: %r", exc)
        duplicate_rows, dup_example_rows = 0, []
    small_sample = total < SMALL_SAMPLE

    data_quality = {
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "duplicate_example_rows": dup_example_rows,
        "small_sample_warning": {
            "triggered": small_sample,
            "row_count": total,
            "threshold": SMALL_SAMPLE,
            "message": (
                f"Only {total} rows after filters — below {SMALL_SAMPLE}; treat every "
                "finding as directional, not conclusive."
                if small_sample else
                f"{total} rows is enough for basic statistics."
            ),
        },
    }

    # ---- summary ------------------------------------------------------------
    try:
        summary = _summary(con.cursor(), table, columns, numeric, dates, texts, where, params, total)
    except Exception as exc:  # noqa: BLE001
        log.warning("insights: summary failed: %r", exc)
        summary = {"rows": total, "columns": len(columns),
                   "column_types": {c["name"]: c["type"] for c in columns},
                   "date_range": None, "key_numbers": {}}

    # If there is nothing to analyse, return early with an honest limitation.
    if total == 0:
        return {
            "summary": summary,
            "data_quality": data_quality,
            "insights": [_insight(
                id="no_data", title="No rows to analyse", category="executive_summary",
                explanation="After applying the current filters, no rows remain, so no "
                            "evidence-backed insight can be computed.",
                why_it_matters="Loosen the filters to bring rows back into scope.",
                confidence=0.0, trust_score=0, is_limitation=True,
                what_to_check_next="Clear or widen the active filters.",
            )],
            "follow_up_questions": ["Which filter is excluding all rows?"],
        }

    # Identifier columns (row ids, sequences, keys) are numeric in type but are
    # not measurements — analyzing them as such yields meaningless correlations
    # and "outliers". Flag once and exclude from those sections (Fix 4). They stay
    # in the schema/stat chips (summary, missing) untouched.
    integer_cols = {c["name"] for c in columns
                    if c["type"] == "number" and "INT" in (c.get("sql_type") or "").upper()}
    try:
        id_cols = _identifier_columns(con.cursor(), table, numeric, integer_cols, where, params, total)
    except Exception as exc:  # noqa: BLE001
        log.warning("insights: identifier detection failed: %r", exc)
        id_cols = set()
    numeric_measures = [c for c in numeric if c not in id_cols]

    facts_exec = _exec_facts(summary, total, numeric, dates, texts)
    insights: list[dict] = []

    # Each section runs isolated: its own cursor, a try/except and a time budget.
    insights += _guarded(con, "Executive summary", "executive_summary", all_names,
        lambda cur: [_exec_card(cur, table, where, params, mode, total, all_names, numeric,
                                dates, texts, summary, missing_frac, facts_exec)])

    findings: list[dict] = []
    findings += _guarded(con, "Data completeness", "missing_data", list(missing_by_column),
        lambda cur: _missing_insights(cur, table, columns, where, params, total, missing_by_column, mode))
    findings += _guarded(con, "Hidden patterns", "hidden_patterns", texts + dates,
        lambda cur: (_concentration_insights(cur, table, texts, where, params, total, missing_frac, mode)
                     + _trend_insights(cur, table, dates, numeric, where, params, total, missing_frac, mode)
                     + _time_pattern_insights(cur, table, dates, texts, where, params, total, missing_frac, mode)))
    findings += _guarded(con, "Correlations", "correlations", numeric_measures,
        lambda cur: _correlation_insights(cur, table, numeric_measures, where, params, missing_frac, mode))
    findings += _guarded(con, "Anomalies", "anomalies", numeric_measures,
        lambda cur: _anomaly_insights(cur, table, numeric_measures, where, params, total, missing_frac, mode))
    findings += _guarded(con, "What changed most", "what_changed_most", dates + texts,
        lambda cur: _what_changed_insights(cur, table, dates, texts, numeric, where, params, missing_frac, mode))

    # top_insights (Fix 2): only real, high-trust findings headline here. Drop
    # nothing-burgers (trust <= TOP_MIN_TRUST), rank by trust then effect size,
    # and if there aren't at least two genuinely strong findings, don't pad the
    # section — say so honestly. Low-trust items still live in their detailed
    # sections below (they're all appended to `insights` afterwards).
    try:
        # Eligible = real findings only: not a limitation, trust above the floor,
        # and a non-zero effect size. The effect-size gate is what makes
        # "nothing-burgers sink or drop": all-clear/descriptive cards ("No column
        # is badly incomplete", "No IQR outliers found") carry high trust but zero
        # notability, so they never headline as a Top insight.
        eligible = sorted(
            [f for f in findings if not f["is_limitation"]
             and f["trust_score"] > TOP_MIN_TRUST and f["_notability"] > 0],
            key=lambda f: (f["trust_score"], f["_notability"]), reverse=True,
        )
        strong = [f for f in eligible if f["trust_score"] > TOP_STRONG_TRUST]
        if len(strong) >= 2:
            for rank, src in enumerate(eligible[:3], start=1):
                top = dict(src)
                top["id"] = f"top_insight_{rank}"
                top["category"] = "top_insights"
                top["title"] = f"#{rank}: {src['title']}"
                top["supporting_metrics"] = {"rank": rank, "source_insight": src["id"],
                                             **src["supporting_metrics"]}
                insights.append(top)
        else:
            insights.append(_insight(
                id="top_insights_none",
                title="No strong patterns found in these columns. Here's what we checked.",
                category="top_insights",
                explanation="No finding cleared the trust bar to headline here. The checks we "
                            "ran — completeness, concentration, trends, correlations, anomalies "
                            "and period-over-period change — appear in their own sections below, "
                            "each with its confidence and trust score, so nothing is overstated.",
                why_it_matters="Promoting weak or low-evidence findings as headlines would claim "
                               "more than the data supports.",
                confidence=0.0, trust_score=0, is_limitation=True,
                evidence_columns=all_names,
                what_to_check_next="Add more rows, cleaner columns, or a clear date/segment to "
                                   "strengthen the analysis.",
                notability=0.0,
            ))
    except Exception as exc:  # noqa: BLE001
        log.warning("insights: ranking top insights failed: %r", exc)

    insights += findings

    try:
        follow_ups = _follow_ups(findings, data_quality, dates, numeric, texts)
    except Exception as exc:  # noqa: BLE001
        log.warning("insights: follow-ups failed: %r", exc)
        follow_ups = ["What question are you hoping this dataset can answer?"]

    return {
        "summary": summary,
        "data_quality": data_quality,
        "insights": insights,
        "follow_up_questions": follow_ups,
    }


def _exec_card(cur, table, where, params, mode, total, all_names, numeric, dates, texts,
               summary, missing_frac, facts_exec) -> dict:
    """The descriptive executive-summary card (effect/consistency = 1)."""
    exec_rows = _rowids(cur, table, where, params, limit=5)
    return _insight(
        id="executive_summary", title="What this dataset is", category="executive_summary",
        explanation=_LEAD_IN[mode] + (
            f"This dataset has {total:,} rows and {len(all_names)} columns "
            f"({len(numeric)} numeric, {len(dates)} date, {len(texts)} text)."
            + (f" It spans {facts_exec['date_span']}." if facts_exec.get("date_span") else "")
            + ("" if total >= SMALL_SAMPLE else
               f" Note: with only {total} rows, confidence is limited.")
        ),
        why_it_matters=_why("executive_summary", mode, facts_exec),
        confidence=_confidence(total, 1.0),
        trust_score=_trust(total, _avg_missing(missing_frac, all_names), 1.0),
        evidence_rows=exec_rows, evidence_columns=all_names,
        supporting_metrics={
            "rows": total, "columns": len(all_names),
            "numeric_columns": numeric, "date_columns": dates, "text_columns": texts,
            "date_range": summary["date_range"], "key_numbers": summary["key_numbers"],
        },
        what_to_check_next="Scan the first rows in the table to sanity-check the columns.",
        notability=0.0,  # never surfaced as a 'top' finding
    )


def _finalize(report: dict) -> dict:
    """Strip internal-only fields before returning to the client."""
    for ins in report.get("insights", []):
        ins.pop("_notability", None)
    return report


# ---------------------------------------------------------------------------
# Data-quality computations.
# ---------------------------------------------------------------------------

def _missing_by_column(cur, table, columns, where, params, total) -> dict[str, dict]:
    """Per-column null (and, for text, empty-string) counts in a single scan."""
    if total == 0:
        return {c["name"]: {"missing": 0, "missing_fraction": 0.0, "type": c["type"]}
                for c in columns}
    parts = []
    for c in columns:
        ident = _qi(c["name"])
        if c["type"] == "text":
            # treat NULL and empty string as missing for text
            parts.append(f"SUM(CASE WHEN {ident} IS NULL OR CAST({ident} AS VARCHAR) = '' THEN 1 ELSE 0 END)")
        else:
            parts.append(f"SUM(CASE WHEN {ident} IS NULL THEN 1 ELSE 0 END)")
    row = cur.execute(f"SELECT {', '.join(parts)} FROM {_qi(table)}{where}", params).fetchone()
    out = {}
    for c, miss in zip(columns, row):
        miss = int(miss or 0)
        out[c["name"]] = {
            "missing": miss,
            "missing_fraction": round(miss / total, 4),
            "missing_pct": round(100 * miss / total, 2),
            "type": c["type"],
        }
    return out


def _duplicates(cur, table, all_names, where, params, total) -> tuple[int, list[int]]:
    """Count fully-duplicate rows and grab a few example rowids of duplicates."""
    if total == 0 or not all_names:
        return 0, []
    distinct = int(cur.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {_qi(table)}{where})", params
    ).fetchone()[0])
    dupes = max(0, total - distinct)
    example: list[int] = []
    if dupes > 0:
        part = ", ".join(_qi(n) for n in all_names)
        # rows whose full tuple has appeared before (rn > 1) are the redundant copies
        sql = (
            f"SELECT rowid FROM (SELECT rowid, ROW_NUMBER() OVER "
            f"(PARTITION BY {part} ORDER BY rowid) rn FROM {_qi(table)}{where}) "
            f"WHERE rn > 1 ORDER BY rowid LIMIT {EVIDENCE_CAP}"
        )
        example = [int(r[0]) for r in cur.execute(sql, params).fetchall()]
    return dupes, example


def _avg_missing(missing_frac: dict[str, float], cols: list[str]) -> float:
    cols = [c for c in cols if c in missing_frac]
    if not cols:
        return 0.0
    return sum(missing_frac[c] for c in cols) / len(cols)


# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------

def _summary(cur, table, columns, numeric, dates, texts, where, params, total) -> dict:
    key_numbers: dict[str, dict] = {}
    for name in numeric[:6]:
        ident = _qi(name)
        s, a, mn, mx = cur.execute(
            f"SELECT SUM({ident}), AVG({ident}), MIN({ident}), MAX({ident}) "
            f"FROM {_qi(table)}{where}", params
        ).fetchone()
        key_numbers[name] = {"sum": _round(s), "avg": _round(a), "min": _round(mn), "max": _round(mx)}

    date_range = None
    for name in dates:
        ident = _qi(name)
        mn, mx = cur.execute(
            f"SELECT MIN({ident}), MAX({ident}) FROM {_qi(table)}{where}", params
        ).fetchone()
        if mn is not None and mx is not None:
            span_days = None
            try:
                span_days = (mx - mn).days
            except Exception:  # noqa: BLE001
                span_days = None
            date_range = {"column": name, "min": _j(mn), "max": _j(mx), "span_days": span_days}
            break

    return {
        "rows": total,
        "columns": len(columns),
        "column_types": {c["name"]: c["type"] for c in columns},
        "date_range": date_range,
        "key_numbers": key_numbers,
    }


def _exec_facts(summary, total, numeric, dates, texts) -> dict:
    dr = summary.get("date_range")
    span = None
    if dr:
        span = f"{dr['min']} to {dr['max']}"
        if dr.get("span_days") is not None:
            span += f" ({dr['span_days']} days)"
    return {"rows": total, "numeric": len(numeric), "dates": len(dates),
            "texts": len(texts), "date_span": span}


# ---------------------------------------------------------------------------
# Missing-data insight.
# ---------------------------------------------------------------------------

def _missing_insights(cur, table, columns, where, params, total, missing_by_column, mode) -> list[dict]:
    flagged = {name: m for name, m in missing_by_column.items()
               if m["missing_fraction"] >= MISSING_FLAG}
    involved = list(flagged.keys())
    if not flagged:
        worst = max(missing_by_column.items(), key=lambda kv: kv[1]["missing_fraction"], default=None)
        worst_pct = worst[1]["missing_pct"] if worst else 0.0
        return [_insight(
            id="missing_data", title="No column is badly incomplete", category="missing_data",
            explanation=f"Your data is fairly complete — no column has a lot of blanks. "
                        f"The gappiest one is only {worst_pct}% empty."
                        + _tech(mode, f" (flag threshold {int(MISSING_FLAG*100)}%)"),
            why_it_matters=_why("missing_data", mode, {}),
            confidence=_confidence(total, 0.5), trust_score=_trust(total, 0.0, 1.0),
            evidence_columns=list(missing_by_column.keys()),
            supporting_metrics={"threshold_pct": int(MISSING_FLAG * 100),
                                "missing_pct_by_column": {k: v["missing_pct"] for k, v in missing_by_column.items()}},
            what_to_check_next="Still confirm that blank cells mean 'unknown' and not 'zero'.",
            is_limitation=False, notability=0.0,
        )]

    # evidence: rows that are missing the worst-offending column
    worst_col = max(flagged, key=lambda c: flagged[c]["missing_fraction"])
    ident = _qi(worst_col)
    extra = (f"({ident} IS NULL OR CAST({ident} AS VARCHAR) = '')"
             if missing_by_column[worst_col]["type"] == "text" else f"{ident} IS NULL")
    ev_rows = _rowids(cur, table, where, params, extra_sql=extra)
    worst_frac = flagged[worst_col]["missing_fraction"]
    return [_insight(
        id="missing_data", title=f"{len(flagged)} column(s) have heavy missing data",
        category="missing_data",
        explanation=f"Some columns have a lot of blanks — {worst_col} is missing "
                    f"{flagged[worst_col]['missing_pct']}% of its values, so any number "
                    f"based on {worst_col} is only partly backed by real data."
                    + ("" if total >= SMALL_SAMPLE else " Sample is small, so treat this as directional.")
                    + _tech(mode, f" ({len(flagged)} column(s) over the {int(MISSING_FLAG*100)}% mark)"),
        why_it_matters=_why("missing_data", mode, {}),
        confidence=_confidence(total, min(1.0, worst_frac / MISSING_FLAG)),
        trust_score=_trust(total, 0.0, 1.0),  # this IS the data-quality claim; report it at full consistency
        evidence_rows=ev_rows, evidence_columns=involved,
        supporting_metrics={
            "threshold_pct": int(MISSING_FLAG * 100),
            "flagged": {c: flagged[c]["missing_pct"] for c in flagged},
            "worst_column": worst_col, "worst_missing_pct": flagged[worst_col]["missing_pct"],
        },
        what_to_check_next=f"Decide whether to impute, drop, or collect more data for {worst_col}.",
        notability=min(1.0, worst_frac) * 0.6,
    )]


# ---------------------------------------------------------------------------
# Concentration / dominant group (hidden pattern).
# ---------------------------------------------------------------------------

def _concentration_insights(cur, table, texts, where, params, total, missing_frac, mode) -> list[dict]:
    out: list[dict] = []
    scored: list[tuple[float, str, str, int, int]] = []  # (share, col, label, count, ndistinct)
    for name in texts:
        ident = _qi(name)
        row = cur.execute(
            f"SELECT CAST({ident} AS VARCHAR) AS label, COUNT(*) c FROM {_qi(table)}{where} "
            f"GROUP BY label ORDER BY c DESC LIMIT 1", params
        ).fetchone()
        if not row or row[0] is None:
            continue
        label, count = row[0], int(row[1])
        share = count / total if total else 0.0
        ndistinct = int(cur.execute(
            f"SELECT COUNT(DISTINCT {ident}) FROM {_qi(table)}{where}", params
        ).fetchone()[0] or 0)
        scored.append((share, name, label, count, ndistinct))

    scored.sort(reverse=True)
    for share, name, label, count, ndistinct in scored[:2]:
        # Fix 3: a column with a single distinct value can't "dominate" or explain
        # anything — reframe it honestly instead of claiming 100% concentration.
        if ndistinct <= 1:
            out.append(_insight(
                id=f"single_value_{name}",
                title=f"'{name}' has only one value ('{label}')",
                category="hidden_patterns",
                explanation=f"Every non-empty row of {name} is '{label}' (1 distinct value), "
                            "so this column can't segment the data or explain any difference.",
                why_it_matters="A constant column adds no analytical signal — it can't split, "
                               "compare or predict anything.",
                confidence=0.0, trust_score=15, is_limitation=True,
                evidence_columns=[name],
                supporting_metrics={"column": name, "distinct_values": 1, "only_value": label},
                what_to_check_next=f"Drop {name} from analysis, or add rows with other values.",
                notability=0.0,
            ))
            continue
        if share < CONCENTRATION_FLAG:
            continue
        ev_rows = _rowids(cur, table, where, params,
                          extra_sql=f"CAST({_qi(name)} AS VARCHAR) = ?", extra_params=[label])
        # Runner-up for comparative context (Part 1): the next-biggest group and its
        # real share, so "dominant" is grounded against the closest rival.
        runner_clause, runner_metrics = ".", {}
        top3 = cur.execute(
            f"SELECT CAST({_qi(name)} AS VARCHAR) AS lab, COUNT(*) c FROM {_qi(table)}{where} "
            f"GROUP BY lab ORDER BY c DESC LIMIT 3", params
        ).fetchall()
        runner = next(((lb, int(c)) for lb, c in top3 if lb is not None and lb != label), None)
        if runner and total:
            second, second_count = runner
            share2 = second_count / total
            runner_clause = f" — more than the next ('{second}', {share2*100:.0f}%)."
            runner_metrics = {"runner_up": second, "runner_up_count": second_count,
                              "runner_up_share_pct": round(share2 * 100, 2)}
        out.append(_insight(
            id=f"concentration_{name}", title=f"'{label}' dominates {name}",
            category="hidden_patterns",
            explanation=f"One group makes up a big share of {name} — '{label}' is "
                        f"{share*100:.0f}% of all rows ({count:,} of {total:,})"
                        + runner_clause
                        + " Your overall numbers mostly describe this group."
                        + ("" if total >= SMALL_SAMPLE else " With a small sample this may not hold.")
                        + _tech(mode, f" ({ndistinct} distinct values, share {share*100:.1f}%)"),
            why_it_matters=_why("concentration", mode, {"label": label, "col": name}),
            confidence=_confidence(total, share),
            trust_score=_trust(total, missing_frac.get(name, 0.0), share),
            evidence_rows=ev_rows, evidence_columns=[name],
            supporting_metrics={"column": name, "dominant_value": label, "count": count,
                                "total": total, "share_pct": round(share * 100, 2),
                                **runner_metrics},
            what_to_check_next=f"Break key metrics down by {name} to separate '{label}' from the rest.",
            notability=share,
        ))
    return out


# ---------------------------------------------------------------------------
# Trend over time (hidden pattern) — linear fit of per-bucket counts.
# ---------------------------------------------------------------------------

def _trend_insights(cur, table, dates, numeric, where, params, total, missing_frac, mode) -> list[dict]:
    if not dates:
        return [_insight(
            id="trend", title="No date column for a trend", category="hidden_patterns",
            explanation="This dataset has no date/time column, so a time trend cannot be computed.",
            why_it_matters=_why("trend", mode, {}), confidence=0.0, trust_score=0,
            is_limitation=True, what_to_check_next="Add or parse a date column to unlock trend analysis.",
            notability=0.0,
        )]
    import numpy as np

    name = dates[0]
    ident = _qi(name)
    # bucket by day; count rows per bucket (only where the date is present)
    w = _augment(where, f"{ident} IS NOT NULL")
    rows = cur.execute(
        f"SELECT date_trunc('day', {ident}) b, COUNT(*) c FROM {_qi(table)}{w} "
        f"GROUP BY b ORDER BY b", params
    ).fetchall()
    if len(rows) < 3:
        return [_insight(
            id="trend", title="Not enough time points for a trend", category="hidden_patterns",
            explanation=f"Only {len(rows)} distinct day-buckets in {name}; a trend needs at least 3.",
            why_it_matters=_why("trend", mode, {}), confidence=0.0, trust_score=0,
            is_limitation=True, evidence_columns=[name],
            what_to_check_next="Use a dataset covering more days to assess a trend.",
            notability=0.0,
        )]

    x = np.arange(len(rows), dtype=float)
    y = np.array([float(r[1]) for r in rows], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else max(0.0, 1 - ss_res / ss_tot)
    first_c, last_c = int(y[0]), int(y[-1])

    # Fix 1: the title must match the math. A slope that rounds to 0 at the
    # displayed precision (2 dp), or a poor fit (R² < STRONG_CORR), is NOT a
    # trend — say "No clear trend over time" and don't claim a direction.
    slope_disp = round(float(slope), 2)
    trend_clear = (r2 >= STRONG_CORR) and (abs(slope_disp) > 0)
    # "very consistent"/"fairly consistent"/"rough" from how well the trend holds.
    steadiness = ("very consistent" if r2 >= 0.8
                  else "fairly consistent" if r2 >= 0.5 else "rough")
    if trend_clear:
        direction = "increasing" if slope > 0 else "decreasing"
        trend_word = "rising" if slope > 0 else "falling"
        title = f"Row volume is {direction} over time"
        plain = (f"Your {name} activity is {trend_word} over time — daily records went "
                 f"from about {first_c:,} to {last_c:,}. It's a {steadiness} trend.")
        next_step = f"Check whether the {direction} trend in {name} is seasonal or a real shift."
    else:
        direction = "flat"
        title = "No clear trend over time"
        plain = (f"Your {name} activity is roughly flat over time — daily records went "
                 f"from about {first_c:,} to {last_c:,}, with no clear direction.")
        next_step = f"Look for seasonality or segment-level movement in {name}; the overall line is flat/noisy."

    # evidence: earliest and latest rows in time
    ev_rows = (_rowids(cur, table, where, params, extra_sql=f"{ident} IS NOT NULL",
                       order_by=f"{ident} ASC", limit=5)
               + _rowids(cur, table, where, params, extra_sql=f"{ident} IS NOT NULL",
                         order_by=f"{ident} DESC", limit=5))
    return [_insight(
        id="trend", title=title, category="hidden_patterns",
        explanation=plain + _tech(mode, f" (slope {slope:+.2f}/day, R²={r2:.2f})"),
        why_it_matters=_why("trend", mode, {}),
        confidence=_confidence(total, r2),
        trust_score=_trust(total, missing_frac.get(name, 0.0), r2),
        evidence_rows=ev_rows, evidence_columns=[name],
        supporting_metrics={"date_column": name, "slope_per_day": round(float(slope), 4),
                            "r_squared": round(r2, 4), "buckets": len(rows),
                            "first_bucket_count": first_c, "last_bucket_count": last_c,
                            "direction": direction},
        what_to_check_next=next_step,
        notability=r2,
    )]


# ---------------------------------------------------------------------------
# Time-of-week / time-of-year rhythm (hidden pattern) — a "so what" interpretation.
# ---------------------------------------------------------------------------

def _time_pattern_insights(cur, table, dates, texts, where, params, total, missing_frac, mode) -> list[dict]:
    """Busiest weekday (and, when the data spans months, the peak month + the
    leading category's own peak day) — Parts 2 and 3.

    Additive and deterministic: emits at most one 'hidden_patterns' card and only
    when a usable date column exists AND the weekday distribution is genuinely
    skewed (top weekday >= 1.3x an even 1/7 share). If the date column is missing,
    the sample is tiny, or the distribution is flat, it emits nothing — never an
    invented pattern. Nulls are excluded and timezone-aware values reduce to their
    wall-clock timestamp (CAST AS TIMESTAMP), so the group-by is always well-typed.
    """
    if not dates:
        return []
    EVEN = 1.0 / 7.0
    SKEW = 1.3  # top day must beat 1.3x an even day to count as a real rhythm
    name = dates[0]
    dident = _qi(name)
    w = _augment(where, f"{dident} IS NOT NULL")

    wd = cur.execute(
        f"SELECT dayname(CAST({dident} AS TIMESTAMP)) AS d, COUNT(*) c "
        f"FROM {_qi(table)}{w} GROUP BY d ORDER BY c DESC", params
    ).fetchall()
    n_dated = sum(int(r[1]) for r in wd)
    if not wd or n_dated < SMALL_SAMPLE or wd[0][0] is None:
        return []
    top_wd, top_wd_c = wd[0][0], int(wd[0][1])
    share = top_wd_c / n_dated
    if share < SKEW * EVEN:
        return []  # essentially flat across the week — invent nothing

    # Optional peak month, only when the data actually spans >= 2 calendar months.
    month_clause, peak_month = "", None
    distinct_months = int(cur.execute(
        f"SELECT COUNT(DISTINCT date_trunc('month', CAST({dident} AS TIMESTAMP))) "
        f"FROM {_qi(table)}{w}", params
    ).fetchone()[0] or 0)
    if distinct_months >= 2:
        mrow = cur.execute(
            f"SELECT monthname(CAST({dident} AS TIMESTAMP)) AS m, COUNT(*) c "
            f"FROM {_qi(table)}{w} GROUP BY m ORDER BY c DESC LIMIT 1", params
        ).fetchone()
        if mrow and mrow[0]:
            peak_month = mrow[0]
            month_clause = f" Activity peaks in {peak_month}."

    # Optional category x weekday (Part 3): the leading category's own peak day,
    # only if that category is itself clearly skewed toward one weekday.
    cat_clause, cat_metrics = "", {}
    if texts:
        tcol = texts[0]
        tid = _qi(tcol)
        lead = cur.execute(
            f"SELECT CAST({tid} AS VARCHAR) AS lab, COUNT(*) c FROM {_qi(table)}{w} "
            f"GROUP BY lab ORDER BY c DESC LIMIT 1", params
        ).fetchone()
        if lead and lead[0] is not None:
            lab = lead[0]
            wc = _augment(w, f"CAST({tid} AS VARCHAR) = ?")
            lwd = cur.execute(
                f"SELECT dayname(CAST({dident} AS TIMESTAMP)) AS d, COUNT(*) c "
                f"FROM {_qi(table)}{wc} GROUP BY d ORDER BY c DESC", [*params, lab]
            ).fetchall()
            n_lab = sum(int(r[1]) for r in lwd)
            if lwd and n_lab >= SMALL_SAMPLE and lwd[0][0] is not None:
                lab_wd, lab_wd_c = lwd[0][0], int(lwd[0][1])
                if lab_wd_c / n_lab >= SKEW * EVEN:
                    cat_clause = f" '{lab}' peaks on {lab_wd}s."
                    cat_metrics = {"category_column": tcol, "leading_category": lab,
                                   "category_peak_weekday": lab_wd}

    ev_rows = _rowids(
        cur, table, where, params,
        extra_sql=f"{dident} IS NOT NULL AND dayname(CAST({dident} AS TIMESTAMP)) = ?",
        extra_params=[top_wd],
    )
    # effect/consistency = how far the busiest day beats an even split (0 at the
    # even line, 1 at double an even day). Scored with the shared helpers.
    effect = max(0.0, min(1.0, share * 7.0 - 1.0))
    return [_insight(
        id="time_pattern", title=f"Busiest on {top_wd}s", category="hidden_patterns",
        explanation=(f"You're busiest on {top_wd}s — about {share*100:.0f}% of all records, "
                     f"noticeably above a typical day.")
                    + month_clause + cat_clause
                    + _tech(mode, f" (even day = {EVEN*100:.0f}%; {top_wd} = {share*100:.1f}%)"),
        why_it_matters=_why("time_pattern", mode, {}),
        confidence=_confidence(n_dated, effect),
        trust_score=_trust(n_dated, missing_frac.get(name, 0.0), effect),
        evidence_rows=ev_rows, evidence_columns=[name],
        supporting_metrics={"date_column": name, "busiest_weekday": top_wd,
                            "busiest_weekday_pct": round(share * 100, 2),
                            "even_day_pct": round(EVEN * 100, 2),
                            "weekday_counts": {str(d): int(c) for d, c in wd},
                            "peak_month": peak_month, "dated_rows": n_dated,
                            **cat_metrics},
        what_to_check_next=f"Check whether the {top_wd} peak is real demand or a data artefact.",
        notability=effect * 0.7,
    )]


# ---------------------------------------------------------------------------
# Correlations — pairwise Pearson on numeric columns.
# ---------------------------------------------------------------------------

def _correlation_insights(cur, table, numeric, where, params, missing_frac, mode) -> list[dict]:
    if len(numeric) < 2:
        return [_insight(
            id="correlation", title="Not enough numeric columns to correlate",
            category="correlations",
            explanation=f"Correlation needs at least 2 numeric columns; this dataset has {len(numeric)}.",
            why_it_matters=_why("correlation", mode, {}), confidence=0.0, trust_score=0,
            is_limitation=True, what_to_check_next="Add numeric columns to explore relationships.",
            notability=0.0,
        )]

    pairs: list[tuple[float, str, str, int]] = []  # (r, a, b, n)
    for a, b in combinations(numeric, 2):
        ia, ib = _qi(a), _qi(b)
        cond = f"{ia} IS NOT NULL AND {ib} IS NOT NULL"
        w = _augment(where, cond)
        r, n = cur.execute(
            f"SELECT corr({ia}, {ib}), COUNT(*) FROM {_qi(table)}{w}", params
        ).fetchone()
        if r is None or n is None or int(n) < MIN_CORR_N:
            continue
        pairs.append((float(r), a, b, int(n)))

    if not pairs:
        return [_insight(
            id="correlation", title="No computable correlations", category="correlations",
            explanation=f"No numeric pair had at least {MIN_CORR_N} rows with both values "
                        "present (and non-constant), so no correlation could be computed.",
            why_it_matters=_why("correlation", mode, {}), confidence=0.0, trust_score=0,
            is_limitation=True, evidence_columns=numeric,
            what_to_check_next="Reduce filtering or fill gaps so numeric pairs overlap.",
            notability=0.0,
        )]

    pairs.sort(key=lambda p: abs(p[0]), reverse=True)
    r, a, b, n = pairs[0]
    # Fix 1: the title must reflect |r|, never claim "move together" for a
    # relationship the number disproves. _corr_relation gives the honest band.
    relation, strength = _corr_relation(r)
    ar = abs(r)
    if ar < WEAK_CORR:
        plain = (f"{a} and {b} don't really move together — knowing one tells you "
                 f"little about the other.")
    else:
        move = "up too" if r >= 0 else "down"
        plain = (f"When {a} goes up, {b} usually goes {move}. It's a {strength} "
                 f"pattern, not proof that one causes the other.")
    ia = _qi(a)
    # evidence: a few lowest-a and highest-a rows (both cols present) to show co-movement
    cond = f"{ia} IS NOT NULL AND {_qi(b)} IS NOT NULL"
    ev_rows = (_rowids(cur, table, where, params, extra_sql=cond, order_by=f"{ia} ASC", limit=4)
               + _rowids(cur, table, where, params, extra_sql=cond, order_by=f"{ia} DESC", limit=4))
    others = [{"pair": [pa, pb], "r": round(pr, 4), "n": pn} for pr, pa, pb, pn in pairs[1:4]]
    return [_insight(
        id="correlation_top", title=f"{a} and {b} are {relation}",
        category="correlations",
        explanation=plain + _tech(mode, f" (r={r:+.2f}, r²={r*r:.2f}, n={n:,})"),
        why_it_matters=_why("correlation", mode, {}),
        confidence=_confidence(n, ar),
        trust_score=_trust(n, (missing_frac.get(a, 0.0) + missing_frac.get(b, 0.0)) / 2, ar),
        evidence_rows=ev_rows, evidence_columns=[a, b],
        supporting_metrics={"pair": [a, b], "pearson_r": round(r, 4), "r_squared": round(r * r, 4),
                            "n": n, "strength": strength, "other_strong_pairs": others},
        what_to_check_next=f"Plot {a} vs {b} and check whether a third variable explains the link.",
        notability=ar,
    )]


# ---------------------------------------------------------------------------
# Anomalies — IQR fence per numeric column.
# ---------------------------------------------------------------------------

def _anomaly_insights(cur, table, numeric, where, params, total, missing_frac, mode) -> list[dict]:
    if not numeric:
        return []
    if total < MIN_OUTLIER_N:
        return [_insight(
            id="anomalies", title="Sample too small for outlier detection",
            category="anomalies",
            explanation=f"Outlier detection needs at least {MIN_OUTLIER_N} rows; only {total} "
                        "are in scope, so any 'outlier' would be unreliable.",
            why_it_matters=_why("anomalies", mode, {}), confidence=0.0, trust_score=0,
            is_limitation=True, what_to_check_next="Gather more rows before flagging outliers.",
            notability=0.0,
        )]

    candidates: list[dict] = []
    for name in numeric:
        ident = _qi(name)
        q1, q3, cnt = cur.execute(
            f"SELECT quantile_cont({ident}, 0.25), quantile_cont({ident}, 0.75), "
            f"COUNT({ident}) FROM {_qi(table)}{where}", params
        ).fetchone()
        if q1 is None or q3 is None or cnt is None or int(cnt) == 0:
            continue
        q1, q3, cnt = float(q1), float(q3), int(cnt)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cond = f"({ident} < {lo} OR {ident} > {hi})"
        n_out = int(cur.execute(
            f"SELECT COUNT(*) FROM {_qi(table)}{_augment(where, cond)}", params
        ).fetchone()[0])
        if n_out == 0:
            continue
        frac = n_out / cnt
        candidates.append({"name": name, "q1": q1, "q3": q3, "iqr": iqr, "lo": lo, "hi": hi,
                           "cnt": cnt, "n_out": n_out, "frac": frac, "cond": cond})

    if not candidates:
        return [_insight(
            id="anomalies", title="No IQR outliers found", category="anomalies",
            explanation="Your numbers are well-behaved — no value stands out as an extreme "
                        "compared with the rest of its column."
                        + _tech(mode, f" (checked {len(numeric)} numeric column(s))"),
            why_it_matters=_why("anomalies", mode, {}),
            confidence=_confidence(total, 0.5), trust_score=_trust(total, 0.0, 1.0),
            evidence_columns=numeric,
            supporting_metrics={"method": "IQR (1.5×) fence", "columns_checked": numeric},
            what_to_check_next="If you expected extremes, verify the column parsed as numeric.",
            notability=0.0,
        )]

    # report up to the 3 columns with the most severe outliers (by fraction)
    candidates.sort(key=lambda c: c["frac"], reverse=True)
    out: list[dict] = []
    for c in candidates[:3]:
        name = c["name"]
        ident = _qi(name)
        ev_rows = _rowids(cur, table, where, params, extra_sql=c["cond"],
                          order_by=f"ABS({ident} - {(c['q1']+c['q3'])/2}) DESC")
        # severity: fraction of rows outside the fence, saturating at 5%
        effect = min(1.0, c["frac"] / 0.05)
        consistency = 1.0 - c["frac"]  # most rows are inside the fence -> consistent bulk
        mn_out, mx_out = cur.execute(
            f"SELECT MIN({ident}), MAX({ident}) FROM {_qi(table)}{_augment(where, c['cond'])}", params
        ).fetchone()
        out.append(_insight(
            id=f"anomaly_{name}", title=f"{c['n_out']:,} outlier row(s) in {name}",
            category="anomalies",
            explanation=f"{c['n_out']:,} value(s) in {name} stand out far from the rest — "
                        f"they range {_round(mn_out,2)}–{_round(mx_out,2)}, while most sit "
                        f"between {_round(c['q1'],2)} and {_round(c['q3'],2)}."
                        + _tech(mode, f" ({c['frac']*100:.1f}% fall outside "
                                      f"[{_round(c['lo'],2)}, {_round(c['hi'],2)}])"),
            why_it_matters=_why("anomalies", mode, {}),
            confidence=_confidence(c["cnt"], effect),
            trust_score=_trust(c["cnt"], missing_frac.get(name, 0.0), consistency),
            evidence_rows=ev_rows, evidence_columns=[name],
            supporting_metrics={"column": name, "method": "IQR 1.5x fence",
                                "q1": _round(c["q1"], 4), "q3": _round(c["q3"], 4),
                                "iqr": _round(c["iqr"], 4), "lower_fence": _round(c["lo"], 4),
                                "upper_fence": _round(c["hi"], 4), "outlier_count": c["n_out"],
                                "values_checked": c["cnt"], "outlier_pct": round(c["frac"] * 100, 2),
                                "min_outlier": _round(mn_out, 4), "max_outlier": _round(mx_out, 4)},
            what_to_check_next=f"Open the {c['n_out']} flagged {name} rows and decide keep/cap/fix.",
            notability=effect * 0.9,
        ))
    return out


# ---------------------------------------------------------------------------
# What changed most — first-half vs second-half by category.
# ---------------------------------------------------------------------------

def _what_changed_insights(cur, table, dates, texts, numeric, where, params, missing_frac, mode) -> list[dict]:
    if not dates:
        return [_insight(
            id="what_changed", title="No date column to compare periods",
            category="what_changed_most",
            explanation="Without a date column the data can't be split into an earlier and "
                        "later half, so period-over-period change can't be computed.",
            why_it_matters=_why("what_changed", mode, {}), confidence=0.0, trust_score=0,
            is_limitation=True, what_to_check_next="Add a date column to see what changed over time.",
            notability=0.0,
        )]
    date_col = dates[0]
    dident = _qi(date_col)
    # epoch(CAST(... AS TIMESTAMP)) makes the split type-agnostic: a DATE column
    # (e.g. order_date) and a TIMESTAMP column both reduce to seconds-since-epoch.
    # This avoids DuckDB's "No function matches +(DATE, DOUBLE)" binder error that
    # naive midpoint arithmetic (MIN + (MAX-MIN)/2) hits on DATE-typed columns.
    eexpr = f"epoch(CAST({dident} AS TIMESTAMP))"
    min_e, max_e = cur.execute(
        f"SELECT MIN({eexpr}), MAX({eexpr}) FROM {_qi(table)}{where}", params
    ).fetchone()
    if min_e is None or max_e is None or float(min_e) == float(max_e):
        return [_insight(
            id="what_changed", title="Date column doesn't span a range",
            category="what_changed_most",
            explanation=f"{date_col} has no usable span (all values equal or null), so an "
                        "earlier-vs-later comparison isn't possible.",
            why_it_matters=_why("what_changed", mode, {}), confidence=0.0, trust_score=0,
            is_limitation=True, evidence_columns=[date_col],
            what_to_check_next="Use a dataset whose dates cover a real time range.",
            notability=0.0,
        )]

    # midpoint split; group by the first text column if present, else overall count
    mid_e = (float(min_e) + float(max_e)) / 2.0
    midpoint = datetime.fromtimestamp(mid_e, tz=timezone.utc).replace(tzinfo=None).isoformat()
    before, after = f"{eexpr} < ?", f"{eexpr} >= ?"
    dim = texts[0] if texts else None

    if dim:
        gid = _qi(dim)
        first = dict(cur.execute(
            f"SELECT CAST({gid} AS VARCHAR), COUNT(*) FROM {_qi(table)}"
            f"{_augment(where, before)} GROUP BY 1", [*params, mid_e]
        ).fetchall())
        second = dict(cur.execute(
            f"SELECT CAST({gid} AS VARCHAR), COUNT(*) FROM {_qi(table)}"
            f"{_augment(where, after)} GROUP BY 1", [*params, mid_e]
        ).fetchall())
        labels = set(first) | set(second)
        # Fix 3: a single group is not a comparison — never claim a "biggest mover
        # across 1 groups". Skip the grouped analysis and fall through to the
        # overall-volume change below (the single-value column is separately
        # flagged by the concentration section).
        if len(labels) > 1:
            movers = []
            for lb in labels:
                f0, s0 = int(first.get(lb, 0)), int(second.get(lb, 0))
                movers.append({"label": lb, "first_half": f0, "second_half": s0, "delta": s0 - f0})
            movers.sort(key=lambda m: abs(m["delta"]), reverse=True)
            n_first = sum(int(v) for v in first.values())
            n_second = sum(int(v) for v in second.values())
            if not movers or (n_first < MIN_HALF_N and n_second < MIN_HALF_N):
                return []
            top = movers[0]
            base = max(1, top["first_half"])
            pct = 100 * top["delta"] / base
            ev_rows = _rowids(cur, table, where, params,
                              extra_sql=f"CAST({gid} AS VARCHAR) = ?", extra_params=[top["label"]])
            effect = min(1.0, abs(top["delta"]) / base)
            # Comparative "now the leader" framing (Part 1): rank groups by total
            # volume across both halves. Only when the biggest MOVER is also the #1
            # group overall do we name the runner-up and the gap — otherwise the
            # "now the top" claim wouldn't be true.
            totals = sorted(
                ((int(first.get(lb, 0)) + int(second.get(lb, 0)), lb) for lb in labels),
                key=lambda t: (t[0], str(t[1])), reverse=True,
            )
            lead_clause, lead_metrics = "", {}
            leader_total, leader_label = totals[0]
            second_total, second_label = totals[1]
            if (leader_label == top["label"] and leader_label not in (None, "")
                    and second_label not in (None, "") and second_total > 0):
                lead_pct = (leader_total - second_total) / second_total * 100
                lead_clause = f" It's now the top {dim}, {lead_pct:.0f}% ahead of '{second_label}'."
                lead_metrics = {"leader_total": leader_total, "second_total": second_total,
                                "second_label": second_label, "lead_pct": round(lead_pct, 2)}
            return [_insight(
                id="what_changed", title=f"'{top['label']}' changed most in {dim}",
                category="what_changed_most",
                explanation=f"'{top['label']}' is the biggest mover in {dim} — it went from "
                            f"{top['first_half']:,} to {top['second_half']:,} ({pct:+.0f}%), the "
                            f"largest change of any {dim}."
                            + lead_clause
                            + _tech(mode, f" ({top['delta']:+,} rows between the two halves)"),
                why_it_matters=_why("what_changed", mode, {}),
                confidence=_confidence(n_first + n_second, effect),
                trust_score=_trust(n_first + n_second, missing_frac.get(dim, 0.0), min(1.0, effect + 0.2)),
                evidence_rows=ev_rows, evidence_columns=[date_col, dim],
                supporting_metrics={"date_column": date_col, "dimension": dim,
                                    "midpoint": midpoint, "top_movers": movers[:6],
                                    "first_half_rows": n_first, "second_half_rows": n_second,
                                    **lead_metrics},
                what_to_check_next=f"Investigate what drove '{top['label']}' between the two halves.",
                notability=effect,
            )]

    # no usable category — report overall volume change between halves
    n_first = int(cur.execute(
        f"SELECT COUNT(*) FROM {_qi(table)}{_augment(where, before)}", [*params, mid_e]
    ).fetchone()[0])
    n_second = int(cur.execute(
        f"SELECT COUNT(*) FROM {_qi(table)}{_augment(where, after)}", [*params, mid_e]
    ).fetchone()[0])
    delta = n_second - n_first
    base = max(1, n_first)
    effect = min(1.0, abs(delta) / base)
    return [_insight(
        id="what_changed", title="Overall volume shifted between halves",
        category="what_changed_most",
        explanation=f"You have {'more' if delta > 0 else 'less'} data in the later half — "
                    f"records went from {n_first:,} to {n_second:,}, {100*delta/base:+.0f}%."
                    + _tech(mode, f" ({delta:+,} rows between halves)"),
        why_it_matters=_why("what_changed", mode, {}),
        confidence=_confidence(n_first + n_second, effect),
        trust_score=_trust(n_first + n_second, missing_frac.get(date_col, 0.0), min(1.0, effect + 0.2)),
        evidence_columns=[date_col],
        supporting_metrics={"date_column": date_col, "midpoint": midpoint,
                            "first_half_rows": n_first, "second_half_rows": n_second, "delta": delta},
        what_to_check_next="Add a category column to see which segment drove the shift.",
        notability=effect,
    )]


# ---------------------------------------------------------------------------
# Follow-up questions — derived from the findings that actually fired.
# ---------------------------------------------------------------------------

def _follow_ups(findings, data_quality, dates, numeric, texts) -> list[str]:
    qs: list[str] = []
    for f in findings:
        if f["is_limitation"]:
            continue
        cat = f["category"]
        m = f["supporting_metrics"]
        if cat == "correlations" and "pair" in m:
            qs.append(f"Does {m['pair'][0]} actually drive {m['pair'][1]}, or is a third factor at play?")
        elif f["id"].startswith("anomaly_"):
            qs.append(f"Are the {m.get('outlier_count')} outliers in {m.get('column')} errors or real extremes?")
        elif f["id"].startswith("concentration_"):
            qs.append(f"Why does '{m.get('dominant_value')}' dominate {m.get('column')}?")
        elif f["id"] == "trend" and m.get("direction") not in (None, "flat"):
            qs.append(f"Is the {m.get('direction')} trend in {m.get('date_column')} seasonal?")
        elif f["id"] == "what_changed" and "dimension" in m:
            qs.append(f"What changed for the top mover in {m.get('dimension')} between the two halves?")
        elif f["id"] == "missing_data" and "worst_column" in m:
            qs.append(f"Why is {m.get('worst_column')} missing so many values?")
    if data_quality["duplicate_rows"] > 0:
        qs.append(f"Should the {data_quality['duplicate_rows']} duplicate rows be de-duplicated?")
    if not qs:
        qs.append("What real-world question are you hoping this dataset can answer?")
    # de-dup while preserving order, cap to 6
    seen, uniq = set(), []
    for q in qs:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq[:6]
