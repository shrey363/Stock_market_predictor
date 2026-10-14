"""News data cleaning and validation.

Rules this step enforces (all of them tested):

* **Traceability.** Every article keeps a derived ``article_id`` (SHA-1 of its
  source URL, or of headline+date when no URL exists), its original headline, its
  source URL, and its publication date.
* **Day-level timestamps are handled conservatively.** The source publishes a
  calendar date, not a time of day, so an article dated ``D`` cannot be proven to
  have existed before a cutoff on ``D``. The cleaned table therefore carries
  ``available_from = published_at + 1 day``; later phases must use that column, not
  ``published_at``, when building features (invariant 2: prediction cutoff).
* **No invented labels.** ``event_category``, ``sentiment_label``,
  ``sentiment_score`` and ``affected_sector`` are left unassigned in Phase 1 and
  reported as such. The source's own ``category`` value is preserved separately as
  ``source_category`` so it is never mistaken for our taxonomy.
* **Duplicates reported.** Duplicates by ``article_id`` and by
  ``(headline, published_at)`` are counted before being removed.
"""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse

import pandas as pd

from ..common.errors import DataSourceError
from ..common.quality import QualityReport

#: Canonical column order of a cleaned news table (spec schema plus documented extras).
NEWS_COLUMNS = [
    "article_id",
    "published_at",
    "available_from",
    "headline",
    "article_text",
    "source",
    "source_url",
    "source_category",
    "authors",
    "event_category",
    "sentiment_label",
    "sentiment_score",
    "affected_sector",
]

#: Not assigned in Phase 1. Left missing on purpose; never filled with a placeholder.
UNASSIGNED_COLUMNS = ("event_category", "sentiment_label", "sentiment_score", "affected_sector")

#: Source columns required before cleaning can proceed.
REQUIRED_SOURCE_COLUMNS = ("headline", "date")


def derive_article_id(link: str | None, headline: str | None, published: str | None) -> str:
    """Stable, derived identifier so an article can be traced back to its raw row."""
    payload = "|".join(part.strip() if isinstance(part, str) else "" for part in (link, headline, published))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def publisher_from_url(url: str | None) -> str | None:
    """Publisher host derived from the article URL, or ``None`` when unobtainable."""
    if not isinstance(url, str) or not url.strip():
        return None
    host = urlparse(url.strip()).netloc.lower()
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


def clean_news_data(
    frame: pd.DataFrame,
    *,
    source_label: str | None = None,
    report: QualityReport | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Validate a raw news table and return ``(cleaned, quality_report)``."""
    quality = report if report is not None else QualityReport(table="news_clean")
    quality.table = "news_clean"
    quality.rows_read = int(len(frame))

    missing_columns = [column for column in REQUIRED_SOURCE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DataSourceError(
            f"news table is missing required column(s) {missing_columns}; present columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise DataSourceError("news table has no rows to clean")

    working = frame.copy()
    working["published_at"] = pd.to_datetime(working["date"], utc=True, errors="coerce", format="ISO8601")
    quality.add_timezone_assumption(
        "the source publishes a calendar date only; published_at is midnight UTC of that date and the "
        "time of day is unknown"
    )

    unparsable = working["published_at"].isna()
    quality.rows_dropped_unparsable_timestamp = int(unparsable.sum())
    if quality.rows_dropped_unparsable_timestamp:
        quality.add_warning(
            f"{quality.rows_dropped_unparsable_timestamp} row(s) had a missing or non-ISO-8601 publication date and were dropped"
        )
        working = working.loc[~unparsable]

    blank_headline = working["headline"].isna() | working["headline"].astype("string").fillna("").str.strip().eq("")
    quality.rows_dropped_missing_required = int(blank_headline.sum())
    if quality.rows_dropped_missing_required:
        quality.add_warning(
            f"{quality.rows_dropped_missing_required} row(s) had no headline and were dropped (a headline is required input)"
        )
        working = working.loc[~blank_headline]

    if working.empty:
        raise DataSourceError("no usable news rows remain after validation")

    # Conservative leakage guard: a day-precision date proves nothing about the time of day.
    working["available_from"] = working["published_at"] + pd.Timedelta(days=1)
    quality.add_note(
        "available_from = published_at + 1 day; use it (not published_at) for feature cutoffs so day-level "
        "timestamps cannot leak same-day, post-cutoff news (invariant 2)"
    )

    working["headline"] = working["headline"].astype("string").str.strip()
    working["source_url"] = working["link"].astype("string") if "link" in working.columns else pd.Series(pd.NA, index=working.index, dtype="string")
    if "link" not in working.columns:
        quality.add_warning("no 'link' column in the raw table; source_url stored as missing")
    if "article_text" not in working.columns:
        working["article_text"] = (
            working["short_description"].astype("string") if "short_description" in working.columns else pd.Series(pd.NA, index=working.index, dtype="string")
        )
        quality.add_note("article_text is the source's short_description field (the dataset has no body text)")
    working["source_category"] = (
        working["category"].astype("string") if "category" in working.columns else pd.Series(pd.NA, index=working.index, dtype="string")
    )
    quality.add_note("source_category is the publisher's own label, kept separate from our event taxonomy")
    working["authors"] = (
        working["authors"].astype("string") if "authors" in working.columns else pd.Series(pd.NA, index=working.index, dtype="string")
    )

    derived_sources = working["source_url"].map(publisher_from_url)
    source_note = "publisher taken from the article URL (host, 'www.' stripped)"
    if source_label:
        working["source"] = derived_sources.fillna(source_label).astype("string")
        source_note += f"; {source_label!r} is used only when a URL is unavailable"
    else:
        working["source"] = derived_sources.astype("string")
        source_note += "; no fallback label is configured, so an article without a URL keeps a missing source"
    quality.add_note(source_note)

    working["article_id"] = [
        derive_article_id(link, headline, published)
        for link, headline, published in zip(
            working["source_url"].astype("string").fillna(""),
            working["headline"].astype("string").fillna(""),
            working["date"].astype("string").fillna(""),
        )
    ]
    quality.add_note("article_id is derived (SHA-1 of source_url|headline|date), not a source field")

    quality.input_chronological = bool(working["published_at"].is_monotonic_increasing)
    if not quality.input_chronological:
        quality.add_warning("raw news rows were not in chronological order; sorted ascending (no shuffling)")
    working = working.sort_values(["published_at", "article_id"], kind="stable")

    quality.duplicate_key = "article_id; then headline+published_at"
    before = len(working)
    working = working.drop_duplicates(subset=["article_id"], keep="first")
    removed_by_id = before - len(working)
    before = len(working)
    working = working.drop_duplicates(subset=["headline", "published_at"], keep="first")
    removed_by_headline = before - len(working)
    quality.duplicates_removed = int(removed_by_id + removed_by_headline)
    quality.add_note(f"duplicates removed: {removed_by_id} identical article_id, {removed_by_headline} identical headline+date")
    if quality.duplicates_removed:
        quality.add_warning(f"{quality.duplicates_removed} duplicate article(s) removed, keeping the first occurrence")

    for column in UNASSIGNED_COLUMNS:
        working[column] = pd.NA
    quality.add_note(
        "event_category, sentiment_label, sentiment_score and affected_sector are intentionally unassigned in "
        "Phase 1 and are reported as missing rather than filled with placeholders"
    )

    working = working[NEWS_COLUMNS].reset_index(drop=True)
    quality.rows_written = int(len(working))
    quality.record_missing(working, NEWS_COLUMNS)
    quality.record_date_range(working["published_at"])
    quality.add_note("no headline text was rewritten, truncated, or normalised beyond trimming surrounding whitespace")
    return working, quality


__all__ = [
    "NEWS_COLUMNS",
    "REQUIRED_SOURCE_COLUMNS",
    "UNASSIGNED_COLUMNS",
    "clean_news_data",
    "derive_article_id",
    "publisher_from_url",
]
