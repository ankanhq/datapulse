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

# Use the small sample file with a tiny cap so the suite is fast and deterministic.
os.environ.setdefault("DATAPULSE_DATA_FILE", "./data_100k.csv")
os.environ.setdefault("DATAPULSE_SAMPLE_ROWS", "3000")

import pytest
from fastapi.testclient import TestClient

import main
import insights as insights_mod

# TestClient only runs the app's lifespan (which opens the DuckDB connection)
# when used as a context manager, so enter it once for the whole session.
client = None


@pytest.fixture(scope="session", autouse=True)
def _app_lifespan():
    global client
    with TestClient(main.app) as c:
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


def test_top_insights_ranked():
    ds = _load_sample()
    rep = client.get(f"/datasets/{ds}/insights").json()
    tops = [i for i in rep["insights"] if i["category"] == "top_insights"]
    assert 1 <= len(tops) <= 3
    ranks = [i["supporting_metrics"]["rank"] for i in tops]
    assert ranks == sorted(ranks) and ranks[0] == 1


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
    lines = ["date,x,y,grp"]
    for i in range(1, 41):
        y = 2 * i
        if i == 20:
            y = 1000  # planted outlier
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
    lines = ["x,y,grp"]
    for i in range(1, 41):
        lines.append(f"{i},{2*i},{'A' if i <= 35 else 'B'}")
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


def test_small_sample_flagged_and_limitations_honest():
    tiny = "a,b\n1,2\n3,4\n5,6\n"  # 3 rows
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

def test_score_formulas_bounds():
    assert insights_mod._confidence(0, 1.0) == 0.0          # no rows -> no confidence
    assert insights_mod._confidence(1000, 1.0) == 1.0       # saturates at 1
    assert insights_mod._confidence(1000, 0.0) == 0.0       # no effect -> no confidence
    assert insights_mod._trust(1000, 0.0, 1.0) == 100       # big, clean, consistent
    assert insights_mod._trust(1000, 1.0, 1.0) == 0         # all missing -> zero trust
    assert 0 <= insights_mod._trust(25, 0.1, 0.5) <= 100
