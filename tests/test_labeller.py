"""Tests for the interactive Phase 2 labeller (`nlp/labeller.py`).

The labeller is the only place in this project that writes a human label, so these
tests pin the safety properties it must keep: labels stay inside the taxonomy,
nothing is written for a skipped row, every change is auditable and undoable, the
sheet survives rewrites with its header and untouched rows intact, and the browser
never sees the classifier's prediction (labelling must stay blind).

Everything runs offline: the server binds loopback on an ephemeral port and is driven
with ``http.client``.
"""

from __future__ import annotations

import csv
import http.client
import json
import threading
from pathlib import Path

import pytest

from stock_project.common.errors import DataNotFetchedError
from stock_project.nlp.labeller import BROWSER_COLUMNS, SheetStore, create_server, main
from stock_project.nlp.taxonomy import EVENT_CATEGORIES

SHEET_COLUMNS = [
    "article_id",
    "published_at",
    "headline",
    "source",
    "source_category",
    "sampling_reason",
    "event_category",
    "reviewer",
    "reviewed_at",
    "notes",
]

SHEET_ROWS = [
    ["a1", "2018-01-01 00:00:00+00:00", "Iran sanctions talks resume", "huffpost.com", "POLITICS", "random", "", "", "", ""],
    ["a2", "2018-01-02 00:00:00+00:00", "Senate committee votes on budget", "huffpost.com", "POLITICS", "random", "", "", "", ""],
    ["a3", "2018-01-03 00:00:00+00:00", "NATO troops deploy near border", "huffpost.com", "POLITICS", "random", "", "", "", ""],
]


def write_sheet(path: Path, rows: list[list[str]] | None = None, columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(columns or SHEET_COLUMNS)
        writer.writerows(rows if rows is not None else SHEET_ROWS)
    return path


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def audit_lines(path: Path) -> list[dict[str, object]]:
    log = path.parent / f"{path.stem}_labelling_log.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def sheet_path(tmp_path: Path) -> Path:
    return write_sheet(tmp_path / "data" / "annotations" / "news_review_sample.csv")


@pytest.fixture
def store(sheet_path: Path) -> SheetStore:
    return SheetStore(sheet_path, reviewer="tester")


@pytest.fixture
def server(store: SheetStore):
    httpd = create_server(store, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def request(server, method: str, path: str, payload: object | None = None) -> tuple[int, bytes]:
    host, port = server.server_address[0], server.server_address[1]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    data = response.read()
    status = response.status
    connection.close()
    return status, data


# -- SheetStore ---------------------------------------------------------------
def test_set_label_writes_only_that_row_and_fills_audit_fields(store: SheetStore, sheet_path: Path) -> None:
    returned = store.set_label("a2", "Sanctions", note="tariffs are trade; embargoes are sanctions")

    assert returned == "sanctions"  # labels are normalised to lower case
    rows = read_sheet(sheet_path)
    labelled = {row["article_id"]: row for row in rows}["a2"]
    assert labelled["event_category"] == "sanctions"
    assert labelled["reviewer"] == "tester"
    assert labelled["reviewed_at"].strip() != ""  # written, but the exact time is not asserted
    assert labelled["notes"] == "tariffs are trade; embargoes are sanctions"
    untouched = [row for row in rows if row["article_id"] != "a2"]
    assert all(row["event_category"] == "" and row["reviewer"] == "" for row in untouched)
    assert len(rows) == 3  # nothing added or dropped


def test_rewrite_preserves_header_and_leaves_no_temporary_file(store: SheetStore, sheet_path: Path) -> None:
    store.set_label("a1", "sanctions")

    assert [line.rstrip("\r\n") for line in sheet_path.read_text(encoding="utf-8").splitlines()][:1] == [
        ",".join(SHEET_COLUMNS)
    ]
    assert not list(sheet_path.parent.glob("*.tmp"))


def test_audit_log_records_every_change(store: SheetStore, sheet_path: Path) -> None:
    store.set_label("a1", "sanctions")
    store.set_label("a1", "armed_conflict", note="changed my mind")
    store.undo()

    entries = audit_lines(sheet_path)
    assert [entry["action"] for entry in entries] == ["label", "label", "undo"]
    assert entries[0]["old_label"] == "" and entries[0]["new_label"] == "sanctions"
    assert entries[1]["old_label"] == "sanctions" and entries[1]["new_label"] == "armed_conflict"
    assert entries[2]["old_label"] == "armed_conflict" and entries[2]["new_label"] == "sanctions"
    assert all(entry["reviewer"] == "tester" for entry in entries)


def test_label_outside_the_taxonomy_is_a_hard_error_and_writes_nothing(store: SheetStore, sheet_path: Path) -> None:
    before = sheet_path.read_bytes()

    with pytest.raises(ValueError) as excinfo:
        store.set_label("a1", "trade war")

    assert "tax" in str(excinfo.value).lower()
    assert sheet_path.read_bytes() == before
    assert audit_lines(sheet_path) == []


def test_unknown_article_id_is_rejected(store: SheetStore) -> None:
    with pytest.raises(KeyError):
        store.set_label("not-in-this-sheet", "sanctions")


def test_undo_restores_the_previous_values(store: SheetStore, sheet_path: Path) -> None:
    store.set_label("a1", "sanctions")
    assert store.undo() is True

    rows = {row["article_id"]: row for row in read_sheet(sheet_path)}
    assert rows["a1"]["event_category"] == "" and rows["a1"]["reviewer"] == ""
    assert store.undo() is False  # nothing left to undo


def test_undo_is_a_no_op_on_an_untouched_sheet(store: SheetStore, sheet_path: Path) -> None:
    before = sheet_path.read_bytes()

    assert store.undo() is False
    assert sheet_path.read_bytes() == before


def test_next_unlabelled_walks_in_order_then_reports_done(store: SheetStore) -> None:
    assert store.next_unlabelled(-1) == 0
    store.set_label("a1", "sanctions")
    assert store.next_unlabelled(-1) == 1

    store.set_label("a2", "other_or_uncertain")
    store.set_label("a3", "armed_conflict")
    assert store.remaining_count() == 0
    assert store.next_unlabelled(-1) is None
    assert store.state()["done"] is True
    assert store.state()["article"] is None
    counts = store.counts()
    assert counts["sanctions"] == 1 and counts["other_or_uncertain"] == 1 and counts["armed_conflict"] == 1
    assert sum(counts.values()) == 3


def test_counts_cover_every_taxonomy_category(store: SheetStore) -> None:
    store.set_label("a1", "sanctions")

    counts = store.counts()
    assert set(counts) == set(EVENT_CATEGORIES)
    assert counts["sanctions"] == 1
    assert sum(counts.values()) == 1


def test_skip_advances_without_writing(store: SheetStore, sheet_path: Path) -> None:
    before = sheet_path.read_bytes()

    store.skip()

    assert sheet_path.read_bytes() == before
    assert audit_lines(sheet_path) == []
    assert store.state()["article"]["article_id"] == "a2"


# -- blindness ----------------------------------------------------------------
def test_browser_payload_and_page_never_expose_predictions(tmp_path: Path) -> None:
    columns = SHEET_COLUMNS[:6] + ["predicted_event_category", "predicted_matched_keywords"] + SHEET_COLUMNS[6:]
    rows = [
        row[:6] + ["sanctions", "zzzkeywordzz"] + row[6:]
        for row in SHEET_ROWS
    ]
    store = SheetStore(write_sheet(tmp_path / "blind.csv", rows=rows, columns=columns), reviewer="tester")
    httpd = create_server(store, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        state_status, state_body = request(httpd, "GET", "/api/state")
        page_status, page_body = request(httpd, "GET", "/")
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert state_status == 200 and page_status == 200
    state_text = state_body.decode("utf-8")
    page_text = page_body.decode("utf-8")
    for leak in ("zzzkeywordzz", "predicted_matched_keywords", "matched_keywords"):
        assert leak not in state_text
        assert leak not in page_text
    article = json.loads(state_text)["article"]
    assert set(article) == set(BROWSER_COLUMNS)


# -- HTTP layer ---------------------------------------------------------------
def test_served_page_keeps_inline_js_quote_escapes(server) -> None:
    # Regression: PAGE is a Python string containing JS single-quoted strings with
    # inline onclick handlers. A lone backslash (choose(\')) is collapsed by Python
    # and served as choose('' — a JS SyntaxError that leaves only the static
    # "Phase 2 review sheet" header visible while /api/state works fine.
    status, body = request(server, "GET", "/")
    assert status == 200
    text = body.decode("utf-8")
    assert "choose(\\'" in text
    assert "act(\\'/api/skip\\')" in text
    assert "act(\\'/api/undo\\')" in text
    assert "choose(''" not in text


def test_http_round_trip_labels_skips_and_undoes(server, sheet_path: Path) -> None:
    status, body = request(server, "POST", "/api/label", {"article_id": "a1", "label": "sanctions", "note": "n"})
    payload = json.loads(body)
    assert status == 200
    assert payload["labelled"] == 1
    assert payload["article"]["article_id"] == "a2"  # advanced to the next unlabelled row
    assert payload["undo_available"] is True
    assert {row["article_id"]: row["event_category"] for row in read_sheet(sheet_path)}["a1"] == "sanctions"

    status, _ = request(server, "POST", "/api/label", {"article_id": "a2", "label": "not_a_category"})
    assert status == 400
    assert all(row["event_category"] != "not_a_category" for row in read_sheet(sheet_path))

    status, body = request(server, "POST", "/api/skip")
    assert status == 200 and json.loads(body)["article"]["article_id"] == "a3"
    assert len(audit_lines(sheet_path)) == 1  # the skip wrote nothing

    status, body = request(server, "POST", "/api/undo")
    assert status == 200 and json.loads(body)["labelled"] == 0
    assert {row["article_id"]: row["event_category"] for row in read_sheet(sheet_path)}["a1"] == ""

    status, _ = request(server, "POST", "/api/undo")
    assert status == 400  # nothing left to undo


def test_http_reports_bad_requests_and_unknown_paths(server) -> None:
    assert request(server, "GET", "/nope")[0] == 404
    assert request(server, "POST", "/nope", {})[0] == 404
    assert request(server, "POST", "/api/label", {"article_id": "a999", "label": "sanctions"})[0] == 400
    assert request(server, "POST", "/api/label", {"article_id": "a1", "label": ""})[0] == 400


def test_create_server_refuses_non_loopback_hosts(store: SheetStore) -> None:
    with pytest.raises(ValueError) as excinfo:
        create_server(store, "0.0.0.0", 0)
    assert "loopback" in str(excinfo.value)


# -- CLI ----------------------------------------------------------------------
def test_check_mode_prints_progress_without_serving(tmp_path: Path, config_path: Path, sheet_path: Path, capsys) -> None:
    exit_code = main(["--config", str(config_path), "--root", str(tmp_path), "--sheet", str(sheet_path), "--check"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "rows      : 3" in output
    assert "remaining : 3" in output


def test_missing_sheet_is_reported_as_not_fetched(tmp_path: Path, config_path: Path) -> None:
    with pytest.raises(DataNotFetchedError):
        main(["--config", str(config_path), "--sheet", str(tmp_path / "nope.csv"), "--check"])


def test_sheet_without_required_columns_is_rejected(tmp_path: Path) -> None:
    path = write_sheet(tmp_path / "broken.csv", columns=["article_id", "headline"])

    with pytest.raises(ValueError) as excinfo:
        SheetStore(path)

    assert "event_category" in str(excinfo.value)


def test_duplicate_article_ids_are_rejected(tmp_path: Path) -> None:
    rows = [list(SHEET_ROWS[0]), list(SHEET_ROWS[0])]

    with pytest.raises(ValueError) as excinfo:
        SheetStore(write_sheet(tmp_path / "dupes.csv", rows=rows))

    assert "duplicate article_id" in str(excinfo.value)
