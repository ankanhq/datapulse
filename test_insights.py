"""Tests for Evidence Mode.

Two layers:
  * structural tests against the bundled sample dataset (shape of the report,
    score ranges, every claim carries numbers), and
  * evidence-integrity tests against a hand-built dataset with a KNOWN outlier,
    a KNOWN strong correlation and a KNOWN dominant category — proving that the
    rows an insight points at actually satisfy the claim (the core rule: never a
    claim without the computation that backs it).

Run:  pytest -q   (from the backend repo root)
"""

import os
import tempfile
import time

# Use the small sample file with a tiny cap so the suite is fast and deterministic.
os.environ.setdefault("DATAPULSE_DATA_FILE", "./data_100k.csv")
os.environ.setdefault("DATAPULSE_SAMPLE_ROWS", "3000")
# Auth + persistence config for tests: a known HS256 secret (so we can mint valid
# tokens) and a throwaway SQLite metadata store.
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-jwt-secret-at-least-32-bytes-long-000")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{tempfile.mkdtemp()}/datapulse_test.db")

import jwt
import pytest
from fastapi.testclient import TestClient

import main
import insights as insights_mod

TEST_USER = "user-aaaa-1111"


def _token(sub=TEST_USER, email="a@example.com"):
    """Mint a valid HS256 Supabase-style access token for a user id."""
    return jwt.encode(
        {"sub": sub, "email": email, "aud": "authenticated", "exp": int(time.time()) + 3600},
        os.environ["SUPABASE_JWT_SECRET"], algorithm="HS256",
    )


# TestClient only runs the app's lifespan (which opens the DuckDB connection)
# when used as a context manager, so enter it once for the whole session.
client = None


@pytest.fixture(scope="session", autouse=True)
def _app_lifespan():
    global client
    with TestClient(main.app) as c:
        # Authenticate every request as TEST_USER by default.
        c.headers.update({"Authorization": f"Bearer {_token()}"})
        client = c
        yield

REQUIRED_INSIGHT_KEYS = {
    "id", "title", "category", "explanation", "why_it_matters", "confidence",
    "trust_score", "evidence_rows", "evidence_columns", "supporting_metrics",
    "what_to_check_next",
}


def _load_sample():
    r = client.post("/datasets/sample")
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


def _paste(csv_text, name="synthetic"):
    r = client.post("/datasets/text", json={"text": csv_text, "name": name})
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


# --------------------------------------------------------------------------
# Structural tests on the real sample dataset.
# --------------------------------------------------------------------------

def test_report_shape_and_scores():
    ds = _load_sample()
    r = client.get(f"/datasets/{ds}/insights", params={"mode": "analyst"})
    assert r.status_code == 200, r.text
    rep = r.json()

    for key in ("summary", "data_quality", "insights", "follow_up_questions"):
        assert key in rep, f"missing top-level key {key}"

    s = rep["summary"]
    assert s["rows"] > 0 and s["columns"] > 0
    assert set(s["column_types"]) and all(v in ("number", "date", "text") for v in s["column_types"].values())

    dq = rep["data_quality"]
    assert "missing_by_column" in dq and "duplicate_rows" in dq and "small_sample_warning" in dq
    assert dq["duplicate_rows"] >= 0

    assert len(rep["insights"]) >= 3
    for ins in rep["insights"]:
        assert REQUIRED_INSIGHT_KEYS <= set(ins), f"insight missing keys: {ins.get('id')}"
        assert 0.0 <= ins["confidence"] <= 1.0
        assert 0 <= ins["trust_score"] <= 100
        assert isinstance(ins["evidence_rows"], list)
        assert all(isinstance(x, int) for x in ins["evidence_rows"])
        # THE core rule: any non-limitation claim must carry the numbers proving it.
        if not ins.get("is_limitation"):
            assert ins["supporting_metrics"], f"{ins['id']} has a claim but no supporting_metrics"


def test_categories_present():
    ds = _load_sample()
    rep = client.get(f"/datasets/{ds}/insights").json()
    cats = {i["category"] for i in rep["insights"]}
    for expected in ("executive_summary", "hidden_patterns", "anomalies",
                     "correlations", "missing_data", "what_changed_most"):
        assert expected in cats, f"category {expected} not produced"


def test_top_insights_ranked(tmp_path):
    # Use the pattern-rich "story" sample so there ARE strong findings to rank.
    # (The uniform-random sample has no real patterns, so it correctly yields the
    # honest "no strong patterns" note instead — see test_top_insights_honest_note.)
    import generate_data

    p = tmp_path / "story.csv"
    generate_data.generate_story(str(p), days=180, seed=7)
    ds = _paste(p.read_text(), name="story")
    rep = client.get(f"/datasets/{ds}/insights").json()
    tops = [i for i in rep["insights"] if i["category"] == "top_insights"]
    assert 1 <= len(tops) <= 3
    ranks = [i["supporting_metrics"]["rank"] for i in tops]
    assert ranks == sorted(ranks) and ranks[0] == 1
    # Fix 2: nothing-burgers never headline — every Top insight beats the trust floor.
    assert all(i["trust_score"] > 20 for i in tops), [(i["title"], i["trust_score"]) for i in tops]


def test_top_insights_honest_note_when_no_strong_pattern():
    # Uniform-random data has no strong pattern; the Top section must say so
    # honestly rather than padding with high-trust "all-clear" non-findings.
    import random

    random.seed(11)
    rows = ["a,b"] + [f"{random.random():.3f},{random.random():.3f}" for _ in range(15)]
    ds = _paste("\n".join(rows))
    rep = client.get(f"/datasets/{ds}/insights").json()
    tops = [i for i in rep["insights"] if i["category"] == "top_insights"]
    assert len(tops) == 1 and tops[0]["is_limitation"] is True
    assert "No strong patterns found" in tops[0]["title"]


def test_audience_mode_changes_words_not_math():
    ds = _load_sample()
    a = client.get(f"/datasets/{ds}/insights", params={"mode": "student"}).json()
    b = client.get(f"/datasets/{ds}/insights", params={"mode": "researcher"}).json()
    a_by_id = {i["id"]: i for i in a["insights"]}
    b_by_id = {i["id"]: i for i in b["insights"]}
    common = set(a_by_id) & set(b_by_id)
    assert common
    worded_differently = False
    for iid in common:
        # identical numbers across audiences
        assert a_by_id[iid]["supporting_metrics"] == b_by_id[iid]["supporting_metrics"]
        assert a_by_id[iid]["confidence"] == b_by_id[iid]["confidence"]
        assert a_by_id[iid]["trust_score"] == b_by_id[iid]["trust_score"]
        if (a_by_id[iid]["why_it_matters"] != b_by_id[iid]["why_it_matters"]
                or a_by_id[iid]["explanation"] != b_by_id[iid]["explanation"]):
            worded_differently = True
    assert worded_differently, "audience mode should change wording"


def test_invalid_mode_falls_back():
    ds = _load_sample()
    rep = client.get(f"/datasets/{ds}/insights", params={"mode": "wizard"}).json()
    assert rep["mode"] == "analyst"


# --------------------------------------------------------------------------
# Evidence-integrity tests on a dataset with KNOWN structure.
# --------------------------------------------------------------------------

def _synthetic_csv():
    # 40 rows: y = 2*x (perfect linear) EXCEPT one planted outlier (row x=20 -> y=1000).
    # grp is "A" for 35 rows and "B" for 5 -> A dominates ~87.5%.
    # y is written as a float so it reads as a continuous MEASUREMENT: identifier
    # detection now (correctly) excludes integer sequences/all-distinct integer
    # columns from outlier analysis, and y is the measurement we're outlier-testing.
    lines = ["date,x,y,grp"]
    for i in range(1, 41):
        y = 2.0 * i
        if i == 20:
            y = 1000.0  # planted outlier
        grp = "A" if i <= 35 else "B"
        lines.append(f"2024-01-{i:02d},{i},{y},{grp}")
    return "\n".join(lines)


def test_anomaly_evidence_rows_are_really_outliers():
    ds = _paste(_synthetic_csv())
    rep = client.get(f"/datasets/{ds}/insights").json()
    anomaly = next((i for i in rep["insights"]
                    if i["id"] == "anomaly_y"), None)
    assert anomaly is not None, "expected an outlier insight for column y"
    lo = anomaly["supporting_metrics"]["lower_fence"]
    hi = anomaly["supporting_metrics"]["upper_fence"]
    assert anomaly["evidence_rows"], "outlier insight must cite rows"

    # Pull the exact cited rows and confirm each really is outside the IQR fence.
    ids = ",".join(str(r) for r in anomaly["evidence_rows"])
    rows = client.get(f"/datasets/{ds}/rows", params={"rowids": ids}).json()["data"]
    assert rows, "rows endpoint returned nothing for cited evidence"
    for row in rows:
        yval = float(row["y"])
        assert yval < lo or yval > hi, f"cited row y={yval} is inside the fence [{lo},{hi}]"
    # the planted y=1000 must be among the flagged values
    assert any(float(r["y"]) == 1000.0 for r in rows)


def _clean_linear_csv():
    # y = 2*x with no outliers -> a (near-)perfect positive correlation.
    # Both are floats so they read as continuous MEASUREMENTS: identifier
    # detection excludes integer id/index/sequence columns from correlation, and
    # here x and y are the two measurements whose relationship we're testing.
    lines = ["x,y,grp"]
    for i in range(1, 41):
        lines.append(f"{float(i)},{float(2*i)},{'A' if i <= 35 else 'B'}")
    return "\n".join(lines)


def test_correlation_is_real_and_strong():
    ds = _paste(_clean_linear_csv())
    rep = client.get(f"/datasets/{ds}/insights").json()
    corr = next((i for i in rep["insights"] if i["id"] == "correlation_top"), None)
    assert corr is not None
    r = corr["supporting_metrics"]["pearson_r"]
    # y is exactly 2*x -> a perfect positive linear correlation.
    assert r > 0.99, f"expected near-perfect correlation, got {r}"
    assert corr["supporting_metrics"]["n"] == 40
    assert corr["evidence_rows"]


def test_concentration_matches_dominant_group():
    ds = _paste(_synthetic_csv())
    rep = client.get(f"/datasets/{ds}/insights").json()
    conc = next((i for i in rep["insights"] if i["id"] == "concentration_grp"), None)
    assert conc is not None, "expected a concentration insight for grp"
    assert conc["supporting_metrics"]["dominant_value"] == "A"
    assert conc["supporting_metrics"]["share_pct"] > 40
    # every cited row should actually belong to the dominant group
    ids = ",".join(str(r) for r in conc["evidence_rows"])
    rows = client.get(f"/datasets/{ds}/rows", params={"rowids": ids}).json()["data"]
    assert rows and all(r["grp"] == "A" for r in rows)


# --------------------------------------------------------------------------
# "So what" interpretation layer: comparative leader framing + time patterns.
# --------------------------------------------------------------------------

def _leader_weekday_csv():
    """A crafted dataset with a CLEAR leader ('Alpha', also #1 overall, surging in
    the later half) and a strong WEDNESDAY skew, spanning multiple months — so the
    leader-framing clause, the runner-up clause and the busiest-weekday insight all
    fire deterministically."""
    import datetime as _dt

    lines = ["date,product"]
    start = _dt.date(2026, 1, 5)  # a Monday
    for wk in range(12):          # 12 weeks -> spans Jan, Feb, Mar
        for dow in range(7):
            d = start + _dt.timedelta(days=wk * 7 + dow)
            alpha_n = 2 if wk < 6 else 6  # Alpha surges in the later half
            for _ in range(alpha_n):
                lines.append(f"{d.isoformat()},Alpha")
            for _ in range(2):            # Beta stays flat
                lines.append(f"{d.isoformat()},Beta")
            if d.weekday() == 2:          # Wednesday spike (all Alpha)
                for _ in range(12):
                    lines.append(f"{d.isoformat()},Alpha")
    return "\n".join(lines)


def test_leader_framing_clause_and_metrics():
    ds = _paste(_leader_weekday_csv(), name="leader_weekday")
    insights = client.get(f"/datasets/{ds}/insights", params={"mode": "founder"}).json()["insights"]

    # Part 1a: the biggest-mover card frames the leader against the runner-up.
    wc = next((i for i in insights if i["id"] == "what_changed"), None)
    assert wc is not None and not wc["is_limitation"]
    m = wc["supporting_metrics"]
    assert m["second_label"] == "Beta"
    assert m["leader_total"] > m["second_total"] > 0
    assert m["lead_pct"] > 0
    assert "now the top" in wc["explanation"] and "Beta" in wc["explanation"]

    # Part 1b: the concentration card names the runner-up with its real share.
    conc = next((i for i in insights if i["id"] == "concentration_product"), None)
    assert conc is not None
    assert conc["supporting_metrics"]["runner_up"] == "Beta"
    assert "more than the next" in conc["explanation"]


def test_busiest_weekday_insight():
    ds = _paste(_leader_weekday_csv(), name="leader_weekday")
    insights = client.get(f"/datasets/{ds}/insights", params={"mode": "founder"}).json()["insights"]

    tp = next((i for i in insights if i["id"] == "time_pattern"), None)
    assert tp is not None and not tp["is_limitation"]
    assert tp["category"] == "hidden_patterns"  # reuses an existing key (no FE change)
    sm = tp["supporting_metrics"]
    assert sm["busiest_weekday"] == "Wednesday"
    assert sm["busiest_weekday_pct"] > sm["even_day_pct"]
    assert sm["peak_month"] is not None  # data spans >= 2 months
    # plain, meaning-first, real weekday name (never an index)
    assert "Wednesday" in tp["explanation"] and "busiest" in tp["explanation"].lower()
    assert tp["evidence_columns"] == ["date"]
    assert tp["evidence_rows"]


def test_time_pattern_technical_clause_only_for_technical_modes():
    ds = _paste(_leader_weekday_csv(), name="leader_weekday")
    founder = client.get(f"/datasets/{ds}/insights", params={"mode": "founder"}).json()["insights"]
    analyst = client.get(f"/datasets/{ds}/insights", params={"mode": "analyst"}).json()["insights"]
    f_tp = next(i for i in founder if i["id"] == "time_pattern")
    a_tp = next(i for i in analyst if i["id"] == "time_pattern")
    # identical numbers across audiences; only the wording changes
    assert f_tp["supporting_metrics"] == a_tp["supporting_metrics"]
    assert f_tp["confidence"] == a_tp["confidence"]
    assert f_tp["trust_score"] == a_tp["trust_score"]
    assert "even day" not in f_tp["explanation"]   # plain audience: no technical clause
    assert "even day" in a_tp["explanation"]        # analyst: one technical clause


def test_flat_weekday_emits_no_time_pattern():
    # Perfectly even weekdays (one row per day) must NOT invent a busiest-weekday card.
    import datetime as _dt

    start = _dt.date(2026, 1, 5)
    lines = ["date,product"]
    for i in range(140):  # 20 full weeks -> each weekday appears exactly 20 times
        d = start + _dt.timedelta(days=i)
        lines.append(f"{d.isoformat()},Alpha")
    ds = _paste("\n".join(lines), name="flat_week")
    insights = client.get(f"/datasets/{ds}/insights").json()["insights"]
    assert not any(i["id"] == "time_pattern" for i in insights)


def test_small_sample_flagged_and_limitations_honest():
    # 3 rows; floats so a/b read as measurements (integer sequences would now be
    # flagged as identifiers and excluded, leaving no column to decline on).
    tiny = "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n"  # 3 rows
    ds = _paste(tiny)
    rep = client.get(f"/datasets/{ds}/insights").json()
    assert rep["data_quality"]["small_sample_warning"]["triggered"] is True
    # with 3 rows, outlier detection must decline rather than invent outliers
    anomaly = next((i for i in rep["insights"] if i["category"] == "anomalies"), None)
    assert anomaly is not None and anomaly["is_limitation"] is True


def test_filters_change_scope():
    ds = _paste(_synthetic_csv())
    full = client.get(f"/datasets/{ds}/insights").json()["summary"]["rows"]
    filt = client.get(
        f"/datasets/{ds}/insights",
        params={"filters": '[{"col":"grp","op":"eq","value":"B"}]'},
    ).json()["summary"]["rows"]
    assert filt == 5 and filt < full


def test_rows_endpoint_preserves_order_and_shape():
    ds = _paste(_synthetic_csv())
    rows = client.get(f"/datasets/{ds}/rows", params={"rowids": "5,1,3"}).json()["data"]
    assert [r["__rowid"] for r in rows] == [5, 1, 3]
    assert all("x" in r and "y" in r for r in rows)


# --------------------------------------------------------------------------
# Direct unit test of the formulas (no HTTP).
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Regression: NON-sample schemas must return HTTP 200 (never 500/503).
# These are the exact shapes that crashed the insights worker before the fix.
# --------------------------------------------------------------------------

def _sales_csv():
    # DATE column named 'order_date' (NOT 'timestamp'); this is the shape that
    # 503'd — DuckDB DATE arithmetic in what_changed_most. 30 rows, 2 numerics.
    rows = ["order_date,product,region,units,revenue"]
    prods, regs = ["Widget", "Gadget", "Gizmo"], ["North", "South", "East", "West"]
    for i in range(30):
        rows.append(f"2024-{1 + i // 28:02d}-{1 + i % 28:02d},{prods[i % 3]},{regs[i % 4]},{i + 1},{(i + 1) * 12.5}")
    return "\n".join(rows)

def _nodate_csv():
    rows = ["product,region,units,revenue"]
    for i in range(30):
        rows.append(f"{'ABC'[i % 3]},{'NS'[i % 2]},{i + 1},{(i + 1) * 3.0}")
    return "\n".join(rows)

def _onlytext_csv():
    rows = ["product,region,status"]
    for i in range(30):
        rows.append(f"{'ABC'[i % 3]},{'NS'[i % 2]},{'ok' if i % 2 else 'bad'}")
    return "\n".join(rows)

def _tiny_csv():
    return "a,b\n1,2\n3,4\n5,6\n"

def _singlecat_csv():
    rows = ["category,uid,amount"]
    for i in range(40):
        rows.append(f"OnlyOne,user_{i},{(i % 10) + 1.5}")
    return "\n".join(rows)

def _messy_csv():
    # duplicate header 'name' + an all-empty column
    rows = ["name,name,empty,val"]
    for i in range(25):
        rows.append(f"x{i},y{i},,{(i % 9) + 1}")
    return "\n".join(rows)


ALL_SCHEMAS = {
    "sales_date_col": _sales_csv,
    "no_date": _nodate_csv,
    "only_text": _onlytext_csv,
    "tiny_3_rows": _tiny_csv,
    "single_category": _singlecat_csv,
    "messy_dupe_headers_nulls": _messy_csv,
}


@pytest.mark.parametrize("name", list(ALL_SCHEMAS))
@pytest.mark.parametrize("mode", ["analyst", "student", "founder", "manager", "researcher"])
def test_insights_never_500s_on_any_schema(name, mode):
    ds = _paste(ALL_SCHEMAS[name]())
    r = client.get(f"/datasets/{ds}/insights", params={"mode": mode})
    assert r.status_code == 200, f"{name}/{mode} -> {r.status_code}: {r.text[:300]}"
    rep = r.json()
    # valid, complete structure
    for key in ("summary", "data_quality", "insights", "follow_up_questions"):
        assert key in rep, f"{name}: missing {key}"
    assert isinstance(rep["insights"], list) and len(rep["insights"]) >= 1
    for ins in rep["insights"]:
        assert REQUIRED_INSIGHT_KEYS <= set(ins), f"{name}: insight missing keys"
        assert 0.0 <= ins["confidence"] <= 1.0
        assert 0 <= ins["trust_score"] <= 100
        assert all(isinstance(x, int) for x in ins["evidence_rows"])
        # critical rule: any *claim* (non-limitation) carries the numbers proving it
        if not ins.get("is_limitation"):
            assert ins["supporting_metrics"], f"{name}: claim without supporting_metrics ({ins['id']})"


def test_sales_date_schema_is_the_regression_and_computes_what_changed():
    # The exact reproducer: a DATE column not named 'timestamp' must not 503, and
    # what_changed_most must actually compute (not just degrade to a limitation).
    ds = _paste(_sales_csv())
    r = client.get(f"/datasets/{ds}/insights", params={"mode": "founder"})
    assert r.status_code == 200
    insights = r.json()["insights"]
    wc = [i for i in insights if i["category"] == "what_changed_most" and not i["is_limitation"]]
    assert wc, "what_changed_most should compute on a real DATE column"
    m = wc[0]["supporting_metrics"]
    assert "midpoint" in m and m["first_half_rows"] + m["second_half_rows"] > 0
    assert wc[0]["evidence_rows"], "the what-changed claim must cite rows"


def test_endpoint_returns_200_even_if_a_section_raises(monkeypatch):
    # Force one section to blow up; the endpoint must still return 200 with a
    # clearly-labelled limitation for that section (never a 500/503).
    import insights as ins_mod

    def boom(*a, **k):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(ins_mod, "_correlation_insights", boom)
    ds = _paste(_sales_csv())
    r = client.get(f"/datasets/{ds}/insights")
    assert r.status_code == 200
    corr = [i for i in r.json()["insights"] if i["category"] == "correlations"]
    assert corr and corr[0]["is_limitation"] is True


def test_story_sample_shows_strong_evidence_cards(tmp_path):
    # The built-in "story" sample must produce strong cards on first click:
    # a confident volume trend, a dominant category, and one obvious outlier.
    import generate_data

    p = tmp_path / "story.csv"
    generate_data.generate_story(str(p), days=180, seed=7)
    ds = _paste(p.read_text(), name="story")
    rep = client.get(f"/datasets/{ds}/insights").json()

    real = [i for i in rep["insights"] if not i.get("is_limitation")]
    hidden = [i for i in real if i["category"] == "hidden_patterns"]
    anomalies = [i for i in real if i["category"] == "anomalies"]

    trend = [i for i in hidden if "over time" in i["title"]]
    assert trend and trend[0]["confidence"] >= 0.7, "expected a confident trend card"

    concentration = [i for i in hidden if "dominates" in i["title"]]
    assert concentration and concentration[0]["confidence"] >= 0.4, "expected a dominant-category card"

    outlier = [i for i in anomalies if "outlier" in i["title"] and "revenue" in i["title"]]
    assert outlier, "expected an outlier card on revenue"
    assert outlier[0]["supporting_metrics"]["outlier_count"] == 1, "expected exactly one obvious outlier"


def test_chart_time_series_split_by(tmp_path):
    # A split-by time series returns one aligned series per category value.
    import generate_data

    p = tmp_path / "story.csv"
    generate_data.generate_story(str(p), days=180, seed=7)
    ds = _paste(p.read_text(), name="story")

    r = client.get(f"/datasets/{ds}/chart", params={
        "chart_type": "time_series", "column": "date", "agg": "avg",
        "y_column": "revenue", "split_by": "product_line", "interval": "month",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["split_by"] == "product_line"
    assert len(body["labels"]) > 0
    assert len(body["series"]) >= 2
    for s in body["series"]:
        assert "key" in s and len(s["values"]) == len(body["labels"])


def test_chart_time_series_without_split_is_unchanged(tmp_path):
    import generate_data

    p = tmp_path / "story.csv"
    generate_data.generate_story(str(p), days=120, seed=3)
    ds = _paste(p.read_text(), name="story")

    r = client.get(f"/datasets/{ds}/chart", params={
        "chart_type": "time_series", "column": "date", "agg": "count", "interval": "month",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body and "series" not in body
    assert all({"time", "value"} <= set(pt) for pt in body["data"])


import contextlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@contextlib.contextmanager
def _csv_server(state):
    """A tiny localhost HTTP server serving CSV from a mutable ``state`` dict
    ({"body": str, "ctype": str}), so tests can exercise fetch + refresh."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = state["body"].encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", state.get("ctype", "text/csv"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # keep test output quiet
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/data.csv"
    finally:
        srv.shutdown()


def test_url_dataset_loads_refreshes_and_supports_evidence(monkeypatch):
    # The local test server is on loopback, which the SSRF guard blocks by design;
    # neutralise just that check so we can exercise the real fetch/parse/refresh.
    monkeypatch.setattr(main, "_reject_if_private", lambda host: None)

    v1 = "date,product,region,units,revenue\n" + "\n".join(
        f"2025-01-{1 + i % 28:02d},Widget,North,{i + 1},{(i + 1) * 10}" for i in range(40)
    )
    state = {"body": v1, "ctype": "text/csv"}
    with _csv_server(state) as url:
        r = client.post("/datasets/url", json={"url": url, "name": "Google Sheet"})
        assert r.status_code == 200, r.text
        body = r.json()
        ds = body["dataset_id"]
        assert body["source"] == "url"
        assert body["source_url"] == url
        assert body["row_count"] == 40
        assert {c["name"] for c in body["columns"]} == {"date", "product", "region", "units", "revenue"}

        # Evidence Mode works on a URL-backed dataset.
        assert client.get(f"/datasets/{ds}/insights").status_code == 200

        # Refresh re-pulls the latest data in place (same id, new row count).
        state["body"] = v1 + "\n" + "\n".join(
            f"2025-02-{1 + i % 28:02d},Gadget,South,{i + 1},{(i + 1) * 20}" for i in range(20)
        )
        r2 = client.post(f"/datasets/{ds}/refresh")
        assert r2.status_code == 200, r2.text
        assert r2.json()["dataset_id"] == ds
        assert r2.json()["row_count"] == 60


def test_url_rejects_private_address():
    # Loopback must be refused by the SSRF guard (no monkeypatch here).
    r = client.post("/datasets/url", json={"url": "http://127.0.0.1:9/x.csv"})
    assert r.status_code == 400
    assert "private" in r.json()["detail"].lower() or "local" in r.json()["detail"].lower()


def test_url_rejects_non_http_scheme():
    r = client.post("/datasets/url", json={"url": "file:///etc/passwd"})
    assert r.status_code == 400


def test_url_rejects_html_response(monkeypatch):
    monkeypatch.setattr(main, "_reject_if_private", lambda host: None)
    state = {"body": "<html><body>not csv</body></html>", "ctype": "text/html"}
    with _csv_server(state) as url:
        r = client.post("/datasets/url", json={"url": url})
        assert r.status_code == 400
        assert "csv" in r.json()["detail"].lower()


def test_refresh_rejects_non_url_dataset():
    ds = _paste("a,b\n1,2\n3,4\n")
    r = client.post(f"/datasets/{ds}/refresh")
    assert r.status_code == 400


def test_ssrf_guard_unwraps_nat64(monkeypatch):
    # On NAT64/IPv6-only networks a public IPv4 host resolves to a 64:ff9b::/96
    # address; the guard must unwrap it and allow the (public) embedded IPv4,
    # while still blocking a private IPv4 reached via NAT64.
    def fake_getaddrinfo(v6):
        return lambda *a, **k: [(10, 1, 6, "", (v6, 0, 0, 0))]

    monkeypatch.setattr(main.socket, "getaddrinfo", fake_getaddrinfo("64:ff9b::808:808"))  # 8.8.8.8
    main._reject_if_private("public.example")  # must not raise

    monkeypatch.setattr(main.socket, "getaddrinfo", fake_getaddrinfo("64:ff9b::a00:5"))  # 10.0.0.5
    with pytest.raises(Exception):
        main._reject_if_private("internal.example")


def test_endpoints_require_a_valid_token():
    # A bare client with no Authorization header must be rejected everywhere.
    anon = TestClient(main.app)
    assert anon.post("/datasets/sample").status_code == 401
    assert anon.get("/datasets").status_code == 401
    assert anon.get("/datasets/whatever/summary").status_code == 401
    # A garbage bearer token is also 401.
    bad = anon.post("/datasets/sample", headers={"Authorization": "Bearer not-a-jwt"})
    assert bad.status_code == 401


def test_datasets_are_private_per_user():
    # User A creates a dataset; user B must not be able to read it (403), and the
    # list endpoint only shows each user their own.
    a_headers = {"Authorization": f"Bearer {_token(sub='owner-A', email='a@x.com')}"}
    b_headers = {"Authorization": f"Bearer {_token(sub='owner-B', email='b@x.com')}"}

    ds = client.post("/datasets/text", json={"text": "x,y\n1,2\n3,4\n", "name": "A-data"},
                     headers=a_headers).json()["dataset_id"]

    assert client.get(f"/datasets/{ds}/summary", headers=a_headers).status_code == 200
    assert client.get(f"/datasets/{ds}/summary", headers=b_headers).status_code == 403
    assert client.get(f"/datasets/{ds}/insights", headers=b_headers).status_code == 403

    a_list = client.get("/datasets", headers=a_headers).json()["datasets"]
    b_list = client.get("/datasets", headers=b_headers).json()["datasets"]
    assert any(d["dataset_id"] == ds for d in a_list)
    assert all(d["dataset_id"] != ds for d in b_list)


def test_reports_are_private_per_user():
    a_headers = {"Authorization": f"Bearer {_token(sub='rep-A')}"}
    b_headers = {"Authorization": f"Bearer {_token(sub='rep-B')}"}
    token = client.post("/reports", json={"report": {"insights": []}, "dataset_name": "d"},
                        headers=a_headers).json()["token"]
    assert client.get(f"/reports/{token}", headers=a_headers).status_code == 200
    assert client.get(f"/reports/{token}", headers=b_headers).status_code == 403


def test_url_dataset_metadata_survives_restart(monkeypatch):
    # A URL-backed dataset must reload from the persistent store after the in-memory
    # registry is cleared (simulating a backend restart), and stay listed.
    monkeypatch.setattr(main, "_reject_if_private", lambda host: None)
    owner = {"Authorization": f"Bearer {_token(sub='persist-user')}"}
    state = {"body": "date,val\n2025-01-01,1\n2025-01-02,2\n2025-01-03,3\n", "ctype": "text/csv"}
    with _csv_server(state) as url:
        ds = client.post("/datasets/url", json={"url": url}, headers=owner).json()["dataset_id"]

        # Simulate a restart: wipe the in-memory registry (metadata stays in the DB).
        main._registry.clear()

        # Still listed (served from the store) ...
        listed = client.get("/datasets", headers=owner).json()["datasets"]
        assert any(d["dataset_id"] == ds for d in listed)
        # ... and reloadable/queryable (re-fetched from the source URL).
        r = client.get(f"/datasets/{ds}/summary", headers=owner)
        assert r.status_code == 200, r.text


def test_jwks_asymmetric_token_is_accepted(monkeypatch):
    # Simulate the NEW asymmetric signing keys: sign an ES256 token and make the
    # JWKS client return its public key, proving non-HS256 verification works.
    from cryptography.hazmat.primitives.asymmetric import ec

    priv = ec.generate_private_key(ec.SECP256R1())
    es_token = jwt.encode({"sub": "ecc-user", "exp": int(time.time()) + 3600},
                          priv, algorithm="ES256")

    class _FakeKey:
        key = priv.public_key()

    monkeypatch.setattr(main.auth, "_get_jwk_client",
                        lambda: type("C", (), {"get_signing_key_from_jwt": lambda self, t: _FakeKey()})())
    assert main.auth.user_id_from_token(es_token) == "ecc-user"


def test_share_report_roundtrip():
    ds = _paste(_synthetic_csv())
    rep = client.get(f"/datasets/{ds}/insights").json()
    created = client.post("/reports", json={"report": rep, "dataset_name": "synthetic"})
    assert created.status_code == 200, created.text
    token = created.json()["token"]
    assert created.json()["url"] == f"/report/{token}"

    got = client.get(f"/reports/{token}")
    assert got.status_code == 200
    stored = got.json()
    assert stored["dataset_name"] == "synthetic"
    assert stored["report"]["summary"]["rows"] == rep["summary"]["rows"]


def test_share_report_bad_token():
    assert client.get("/reports/not-a-real-token").status_code == 400
    assert client.get("/reports/" + "a" * 32).status_code == 404


def test_score_formulas_bounds():
    assert insights_mod._confidence(0, 1.0) == 0.0          # no rows -> no confidence
    assert insights_mod._confidence(1000, 1.0) == 1.0       # saturates at 1
    assert insights_mod._confidence(1000, 0.0) == 0.0       # no effect -> no confidence
    assert insights_mod._trust(1000, 0.0, 1.0) == 100       # big, clean, consistent
    assert insights_mod._trust(1000, 1.0, 1.0) == 0         # all missing -> zero trust
    assert 0 <= insights_mod._trust(25, 0.1, 0.5) <= 100
