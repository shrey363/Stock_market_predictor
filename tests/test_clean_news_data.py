"""News cleaning: traceability, day-precision cutoffs, duplicates, unassigned labels."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_project.common.errors import DataSourceError
from stock_project.preprocessing.clean_news_data import (
    NEWS_COLUMNS,
    UNASSIGNED_COLUMNS,
    clean_news_data,
    derive_article_id,
    publisher_from_url,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(**overrides) -> dict:
    row = {
        "link": "https://www.example.com/a",
        "headline": "Headline A",
        "category": "POLITICS",
        "short_description": "Body A",
        "authors": "Author A",
        "date": "2020-01-02",
    }
    row.update(overrides)
    return row


def test_fixture_counts_are_exact(news_subset_raw: pd.DataFrame):
    cleaned, quality = clean_news_data(news_subset_raw)
    assert quality.rows_read == 6
    assert quality.rows_dropped_unparsable_timestamp == 1
    assert quality.rows_dropped_missing_required == 1
    assert quality.duplicates_removed == 2
    assert quality.rows_written == 2
    assert len(cleaned) == 2
    assert "article_id; then headline+published_at" == quality.duplicate_key


def test_article_ids_are_derived_unique_and_stable(news_subset_raw: pd.DataFrame):
    cleaned, quality = clean_news_data(news_subset_raw)
    assert cleaned["article_id"].is_unique
    assert len(cleaned["article_id"].iloc[0]) == 16
    again, _ = clean_news_data(news_subset_raw)
    assert list(again["article_id"]) == list(cleaned["article_id"])
    assert any("derived" in note for note in quality.notes)


def test_derive_article_id_depends_on_inputs():
    assert derive_article_id("https://a", "H", "2020-01-02") == derive_article_id("https://a", "H", "2020-01-02")
    assert derive_article_id("https://a", "H", "2020-01-02") != derive_article_id("https://b", "H", "2020-01-02")


def test_day_precision_is_documented_and_available_from_is_next_day(news_subset_raw: pd.DataFrame):
    cleaned, quality = clean_news_data(news_subset_raw)
    row = cleaned.iloc[0]
    assert row["published_at"] == pd.Timestamp("2020-01-02T00:00:00Z")
    assert row["available_from"] == pd.Timestamp("2020-01-03T00:00:00Z")
    assert ((cleaned["available_from"] - cleaned["published_at"]) == pd.Timedelta(days=1)).all()
    assert any("calendar date only" in note for note in quality.timezone_assumptions)
    assert any("available_from" in note for note in quality.notes)


def test_labels_are_never_invented(news_subset_raw: pd.DataFrame):
    cleaned, quality = clean_news_data(news_subset_raw)
    for column in UNASSIGNED_COLUMNS:
        assert cleaned[column].isna().all(), column
        assert quality.missing_counts[column] == len(cleaned)
    assert quality.imputation_applied == []
    assert any("intentionally unassigned" in note for note in quality.notes)


def test_source_is_derived_from_the_url_without_www(news_subset_raw: pd.DataFrame):
    cleaned, quality = clean_news_data(news_subset_raw)
    assert set(cleaned["source"]) == {"example.com"}
    assert any("publisher taken from the article URL" in note for note in quality.notes)
    assert all(publisher_from_url(url) == "example.com" for url in cleaned["source_url"])


def test_publisher_fallback_is_only_used_when_no_url_exists():
    frame = _frame([_row(link=None)])
    cleaned, quality = clean_news_data(frame, source_label="example dataset")
    assert cleaned.loc[0, "source"] == "example dataset"
    assert cleaned.loc[0, "source_url"] is pd.NA or pd.isna(cleaned.loc[0, "source_url"])
    assert any("used only when a URL is unavailable" in note for note in quality.notes)


def test_article_text_comes_from_short_description():
    cleaned, quality = clean_news_data(_frame([_row(short_description="Body text here")]))
    assert cleaned.loc[0, "article_text"] == "Body text here"
    assert any("short_description" in note for note in quality.notes)


def test_source_category_is_kept_separate_from_our_taxonomy():
    cleaned, _ = clean_news_data(_frame([_row(category="POLITICS")]))
    assert cleaned.loc[0, "source_category"] == "POLITICS"
    assert pd.isna(cleaned.loc[0, "event_category"])


def test_blank_and_missing_headlines_count_as_missing():
    frame = _frame([_row(headline=""), _row(link="https://www.example.com/b", headline=None), _row(link="https://www.example.com/c")])
    cleaned, quality = clean_news_data(frame)
    assert quality.rows_dropped_missing_required == 2
    assert len(cleaned) == 1


def test_output_is_chronological_and_matches_the_column_contract():
    frame = _frame(
        [
            _row(link="https://www.example.com/late", headline="Late", date="2020-01-05"),
            _row(link="https://www.example.com/early", headline="Early", date="2020-01-02"),
        ]
    )
    cleaned, quality = clean_news_data(frame)
    assert quality.input_chronological is False
    assert list(cleaned["published_at"].dt.date.astype(str)) == ["2020-01-02", "2020-01-05"]
    assert list(cleaned.columns) == NEWS_COLUMNS


def test_missing_required_column_raises():
    frame = _frame([_row()]).drop(columns=["date"])
    with pytest.raises(DataSourceError, match="date"):
        clean_news_data(frame)


def test_empty_table_raises():
    with pytest.raises(DataSourceError, match="no rows"):
        clean_news_data(_frame([_row()]).iloc[0:0])


def test_unparsable_dates_are_reported_not_dropped_silently():
    frame = _frame([_row(link="https://www.example.com/x", date="2020-13-45")])
    with pytest.raises(DataSourceError, match="no usable news rows"):
        clean_news_data(frame)


def test_report_serialises_with_subset_quality_numbers():
    frame = _frame([_row()])
    _, quality = clean_news_data(frame)
    payload = quality.to_dict()
    assert payload["date_range"] == ["2020-01-02T00:00:00Z", "2020-01-02T00:00:00Z"]
    assert payload["missing_counts"]["event_category"] == 1
