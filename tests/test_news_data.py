"""News ingestion: provenance, streaming filters, and deterministic subsampling.

The downloader is always a fake writer, so no test performs a network request.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.errors import DataSourceError
from stock_project.common.manifest import read_manifest, sha256_file
from stock_project.data.news_data import (
    SOURCE_COLUMNS,
    extract_news_subset,
    fetch_news_dataset,
    iter_jsonl,
    load_news_subset,
    raw_news_path,
    resolve_url,
    subset_path,
    write_news_subset,
)

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE_FIXTURE = FIXTURES / "news_source.jsonl"

WINDOW = (date(2020, 1, 1), date(2020, 1, 6))


def _copy_fixture_downloader(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(SOURCE_FIXTURE.read_bytes())


def test_resolve_url_pins_the_revision():
    url = resolve_url("example/news", "data.json", "abc123")
    assert "example/news" in url and "abc123" in url and url.endswith("data.json")


def test_fetch_news_dataset_records_provenance(tmp_path: Path):
    path = fetch_news_dataset(
        "example/news",
        "data.json",
        revision="abc123",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        root=tmp_path,
        downloader=_copy_fixture_downloader,
    )
    assert path == raw_news_path("example/news", "data.json", root=tmp_path)
    entry = read_manifest(tmp_path)["datasets"]["news:example/news"]
    assert entry["revision"] == "abc123"
    assert entry["license"] == "CC BY 4.0"
    assert entry["license_url"].startswith("https://creativecommons.org/")
    assert "abc123" in entry["url"]
    assert entry["sha256"] == sha256_file(path)
    assert entry["retrieved_at_utc"].endswith("Z")


def test_fetch_news_dataset_reuses_the_cached_file(tmp_path: Path):
    calls: list[str] = []

    def counting_downloader(url: str, destination: Path) -> None:
        calls.append(url)
        _copy_fixture_downloader(url, destination)

    fetch_news_dataset("example/news", "data.json", revision="abc123", root=tmp_path, downloader=counting_downloader)
    fetch_news_dataset("example/news", "data.json", revision="abc123", root=tmp_path, downloader=counting_downloader)
    assert len(calls) == 1


def test_iter_jsonl_skips_blank_lines_and_reports_malformed_ones():
    records = list(iter_jsonl(SOURCE_FIXTURE))
    assert len(records) == 8
    assert sum(1 for _line, record in records if not record) == 1


def test_extract_subset_counts_are_exact():
    frame = extract_news_subset(SOURCE_FIXTURE, category="POLITICS", start=WINDOW[0], end=WINDOW[1])
    assert len(frame) == 3
    assert list(frame.columns) == SOURCE_COLUMNS
    assert list(frame["date"]) == ["2020-01-02", "2020-01-05", "2020-01-05"]


def test_extract_subset_reports_what_it_skipped():
    from stock_project.common.quality import QualityReport

    quality = QualityReport(table="unset")
    extract_news_subset(SOURCE_FIXTURE, category="POLITICS", start=WINDOW[0], end=WINDOW[1], report=quality)
    assert quality.rows_read == 8
    assert quality.rows_written == 3
    assert quality.rows_outside_window == 2
    assert quality.rows_dropped_missing_required == 1
    assert any("different source category: 1" in note for note in quality.notes)
    assert any("unparsable" in warning for warning in quality.warnings)
    assert any("no publication date" in warning for warning in quality.warnings)
    assert quality.date_range == ("2020-01-02T00:00:00Z", "2020-01-05T00:00:00Z")
    assert quality.to_dict()["date_range"] == ["2020-01-02T00:00:00Z", "2020-01-05T00:00:00Z"]


def test_extract_subset_category_filter_is_case_insensitive():
    upper = extract_news_subset(SOURCE_FIXTURE, category="POLITICS", start=WINDOW[0], end=WINDOW[1])
    lower = extract_news_subset(SOURCE_FIXTURE, category="politics", start=WINDOW[0], end=WINDOW[1])
    assert list(upper["link"]) == list(lower["link"])


def test_cap_is_deterministic_and_spreads_coverage(tmp_path: Path):
    source = tmp_path / "daily.jsonl"
    lines = [
        json.dumps(
            {
                "link": f"https://www.example.com/{day}",
                "headline": f"Headline {day}",
                "category": "POLITICS",
                "short_description": "Body",
                "authors": "Author",
                "date": f"2020-01-{day:02d}",
            }
        )
        for day in range(1, 11)
    ]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    first = extract_news_subset(source, category="POLITICS", max_articles=4)
    second = extract_news_subset(source, category="POLITICS", max_articles=4)
    assert len(first) == 4
    assert list(first["link"]) == list(second["link"])
    days = list(pd.to_datetime(first["date"]).dt.day)
    assert days == sorted(days)
    assert days[0] == 1
    assert len(set(days)) == 4


def test_no_cap_keeps_everything_in_window():
    frame = extract_news_subset(SOURCE_FIXTURE, category="POLITICS", start=WINDOW[0], end=WINDOW[1], max_articles=None)
    assert len(frame) == 3


def test_subset_round_trips_through_disk(tmp_path: Path):
    frame = extract_news_subset(SOURCE_FIXTURE, category="POLITICS", start=WINDOW[0], end=WINDOW[1])
    target = subset_path("POLITICS", *WINDOW, root=tmp_path)
    write_news_subset(frame, target)
    assert target.name == "news_politics_20200101_20200106.csv"
    loaded = load_news_subset(target)
    assert list(loaded.columns) == SOURCE_COLUMNS
    assert len(loaded) == 3


def test_missing_inputs_raise_actionable_errors(tmp_path: Path):
    with pytest.raises(DataSourceError, match="raw news file not found"):
        extract_news_subset(tmp_path / "absent.jsonl")
    with pytest.raises(DataSourceError, match="news subset file not found"):
        load_news_subset(tmp_path / "absent.csv")
