"""News dataset ingestion.

The Phase 1 news source is the Hugging Face mirror of the Kaggle "News Category
Dataset" (HuffPost headlines, CC BY 4.0 — attribution required). See
``docs/dataset_notes.md`` for the full provenance record.

Two-step design, because the upstream file is large (~87 MB JSON Lines):

1. :func:`fetch_news_dataset` downloads the pinned revision into ``data/raw/news/``
   (git-ignored, hashed into the manifest) and *never* edits it.
2. :func:`extract_news_subset` streams that file and writes a small, filtered CSV
   under ``data/interim/``: rows of one source category inside the configured
   date window. If more rows match than the configured cap, a deterministic,
   evenly spaced subsample is taken so coverage stays spread over the window
   instead of collapsing onto the most recent weeks.

The downloader and the raw path are injectable, so the test suite runs offline
against a tiny fixture file.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path
from typing import Callable, Iterator, Mapping

import pandas as pd

from ..common.errors import DataSourceError
from ..common.logging_utils import get_logger
from ..common.manifest import make_dataset_entry, write_manifest
from ..common.paths import interim_dir, raw_news_dir, relative_to_root
from ..common.quality import QualityReport

logger = get_logger(__name__)

#: Columns as published by the source dataset, in their published order.
SOURCE_COLUMNS = ["link", "headline", "category", "short_description", "authors", "date"]

HF_RESOLVE_TEMPLATE = "https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{source_file}"

#: ``downloader(url, destination) -> None``
Downloader = Callable[[str, Path], None]


def dataset_slug(dataset_id: str) -> str:
    return dataset_id.strip("/").split("/")[-1]


def raw_news_path(dataset_id: str, source_file: str, *, root: Path | str | None = None) -> Path:
    return raw_news_dir(root) / dataset_slug(dataset_id) / source_file


def subset_filename(category: str | None, start: date, end: date) -> str:
    label = (category or "all").strip().lower().replace(" ", "_") or "all"
    return f"news_{label}_{start:%Y%m%d}_{end:%Y%m%d}.csv"


def subset_path(
    category: str | None,
    start: date,
    end: date,
    *,
    root: Path | str | None = None,
) -> Path:
    return interim_dir(root) / subset_filename(category, start, end)


def resolve_url(dataset_id: str, source_file: str, revision: str) -> str:
    return HF_RESOLVE_TEMPLATE.format(dataset_id=dataset_id, revision=revision, source_file=source_file)


def _stream_download(url: str, destination: Path, *, timeout: int = 120, chunk_size: int = 1 << 20) -> None:
    """Stream a URL to disk, reporting progress at ~25% steps."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "stock-project/0.1 (research prototype)"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            next_report = 0.25
            while chunk := response.read(chunk_size):
                handle.write(chunk)
                written += len(chunk)
                if total and written / total >= next_report:
                    logger.info("downloaded %.0f%% of %.1f MB", 100 * written / total, total / 1e6)
                    next_report += 0.25
    except Exception as exc:
        raise DataSourceError(f"failed to download {url}: {exc}") from exc
    if not destination.exists() or destination.stat().st_size == 0:
        raise DataSourceError(f"download of {url} produced an empty file at {destination}")


def fetch_news_dataset(
    dataset_id: str,
    source_file: str,
    *,
    revision: str,
    dataset_url: str | None = None,
    license_name: str | None = None,
    license_url: str | None = None,
    root: Path | str | None = None,
    downloader: Downloader | None = None,
    force: bool = False,
) -> Path:
    """Download (or reuse) the raw news file and record its provenance."""
    target = raw_news_path(dataset_id, source_file, root=root)
    if target.exists() and not force:
        logger.info("reusing cached news source: %s", relative_to_root(target, root))
        return target

    url = resolve_url(dataset_id, source_file, revision)
    fetch = downloader or _stream_download
    logger.info("downloading %s", url)
    fetch(url, target)

    entry_key = f"news:{dataset_id}"
    write_manifest(
        root,
        key=entry_key,
        entry=make_dataset_entry(
            key=entry_key,
            name=f"{dataset_id} ({source_file})",
            file_path=target,
            root=root,
            source=dataset_url or f"https://huggingface.co/datasets/{dataset_id}",
            url=url,
            dataset_id=dataset_id,
            revision=revision,
            license=license_name,
            license_url=license_url,
            notes=[
                "Raw download, stored unmodified; the subset used by the project is filtered in data/interim/.",
                "JSON Lines: one JSON object per line.",
            ],
        ),
    )
    return target


def iter_jsonl(path: Path | str) -> Iterator[tuple[int, dict]]:
    """Yield ``(line_number, object)`` for each parsable JSON-lines record."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                yield (line_number, {})
                continue
            if isinstance(record, Mapping):
                yield (line_number, dict(record))


def _even_subsample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Deterministic evenly spaced subsample that preserves coverage of the window."""
    if limit <= 0 or len(frame) <= limit:
        return frame
    step = len(frame) / limit
    indices = sorted({min(len(frame) - 1, int(round(i * step))) for i in range(limit)})
    return frame.iloc[indices].reset_index(drop=True)


def extract_news_subset(
    raw_path: Path | str,
    *,
    category: str | None = None,
    start: date | None = None,
    end: date | None = None,
    max_articles: int | None = None,
    report: QualityReport | None = None,
) -> pd.DataFrame:
    """Filter the raw JSON-lines file into a small, chronologically sorted table."""
    source = Path(raw_path)
    if not source.exists():
        raise DataSourceError(f"raw news file not found: {source}")

    quality = report if report is not None else QualityReport(table="news_subset")
    wanted = category.strip().upper() if category else None
    records: list[dict] = []
    malformed = 0
    outside_window = 0
    wrong_category = 0
    missing_date = 0
    lines_read = 0

    for _line_number, record in iter_jsonl(source):
        lines_read += 1
        if not record:
            malformed += 1
            continue
        if wanted and str(record.get("category", "")).strip().upper() != wanted:
            wrong_category += 1
            continue
        raw_date = record.get("date")
        if raw_date in (None, ""):
            missing_date += 1
            continue
        try:
            published = date.fromisoformat(str(raw_date).strip()[:10])
        except ValueError:
            malformed += 1
            continue
        if start is not None and published < start:
            outside_window += 1
            continue
        if end is not None and published > end:
            outside_window += 1
            continue
        records.append(
            {
                "link": record.get("link"),
                "headline": record.get("headline"),
                "category": record.get("category"),
                "short_description": record.get("short_description"),
                "authors": record.get("authors"),
                "date": published.isoformat(),
            }
        )

    frame = pd.DataFrame(records, columns=SOURCE_COLUMNS)
    frame = frame.sort_values(["date", "link"], kind="stable").reset_index(drop=True)
    before_cap = len(frame)
    if max_articles is not None and len(frame) > max_articles:
        frame = _even_subsample(frame, max_articles)

    quality.table = "news_subset"
    quality.rows_read = lines_read
    quality.rows_written = int(len(frame))
    quality.rows_outside_window = outside_window
    quality.rows_dropped_missing_required = missing_date
    quality.duplicate_key = ""
    quality.notes.extend(
        [
            f"source category filter: {wanted or 'none'}",
            f"date window (inclusive): {start.isoformat() if start else 'begin'}..{end.isoformat() if end else 'end'}",
            f"rows in window before the {max_articles} cap: {before_cap}",
            f"rows skipped for a different source category: {wrong_category}",
        ]
    )
    if before_cap > (max_articles or before_cap):
        quality.add_note("deterministic evenly spaced subsample applied to respect max_articles")
    if malformed:
        quality.add_warning(f"{malformed} line(s) were unparsable or had an invalid date and were skipped")
    if missing_date:
        quality.add_warning(f"{missing_date} record(s) had no publication date and were skipped (never guessed)")
    if len(frame):
        quality.record_date_range(pd.to_datetime(frame["date"], utc=True, errors="coerce"))
    return frame


def write_news_subset(frame: pd.DataFrame, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, lineterminator="\n")
    return target


def load_news_subset(path: Path | str) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        raise DataSourceError(f"news subset file not found: {target}")
    return pd.read_csv(target)


__all__ = [
    "HF_RESOLVE_TEMPLATE",
    "SOURCE_COLUMNS",
    "Downloader",
    "dataset_slug",
    "extract_news_subset",
    "fetch_news_dataset",
    "iter_jsonl",
    "load_news_subset",
    "raw_news_path",
    "resolve_url",
    "subset_filename",
    "subset_path",
    "write_news_subset",
]
