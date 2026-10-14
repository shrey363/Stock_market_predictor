"""Market data ingestion.

Fetches daily OHLCV bars from a remote provider into ``data/raw/market/`` and
records provenance in ``data/manifest.json``.

Design notes that matter for later phases:

* The downloader is injectable, so the test suite exercises caching, the
  manifest, and column normalisation with a fake provider and **no network**.
* Raw files are written exactly as normalised but never edited afterwards; every
  cleaning step produces a new file under ``data/interim`` or ``data/processed``.
* A cached file is reused unless ``force=True``, which keeps re-runs cheap and
  reproducible.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from ..common.errors import DataSourceError
from ..common.logging_utils import get_logger
from ..common.manifest import make_dataset_entry, write_manifest
from ..common.paths import raw_market_dir, relative_to_root

logger = get_logger(__name__)

#: Canonical column order for a cached market table.
MARKET_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "adj_close", "volume", "source"]

#: Where the data came from. Recorded per row so a later merge cannot lose provenance.
DEFAULT_SOURCE_LABEL = "Yahoo Finance via yfinance"

#: ``downloader(symbol, start, end, interval) -> DataFrame``
Downloader = Callable[[str, date, date, str], pd.DataFrame]


def market_filename(symbol: str, interval: str, start: date, end: date) -> str:
    return f"{symbol.upper()}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.csv"


def market_file_path(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    *,
    root: Path | str | None = None,
) -> Path:
    return raw_market_dir(root) / market_filename(symbol, interval, start, end)


def yfinance_downloader(symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """Default downloader. Requires network access and the optional ``yfinance`` package."""
    try:
        import yfinance  # imported lazily: tests and offline steps must not need it
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DataSourceError(
            "yfinance is not installed. Install it (pip install -r requirements.txt) or pass an "
            "explicit downloader when a network fetch is not wanted."
        ) from exc

    try:
        frame = yfinance.Ticker(symbol).history(
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval,
            auto_adjust=False,
            actions=False,
        )
    except Exception as exc:  # network/httpx/provider errors surface as many types
        raise DataSourceError(f"market download failed for {symbol} ({start}..{end}): {exc}") from exc
    if frame is None or frame.empty:
        raise DataSourceError(
            f"market download returned no rows for {symbol} ({start}..{end}, interval={interval}). "
            "This is usually a provider rate limit or an invalid symbol."
        )
    return frame


def normalise_download(frame: pd.DataFrame, *, symbol: str, source_label: str = DEFAULT_SOURCE_LABEL) -> pd.DataFrame:
    """Map a provider frame onto :data:`MARKET_COLUMNS` without inventing values."""
    if frame.empty:
        raise DataSourceError("cannot normalise an empty market frame")
    working = frame.copy()
    if not isinstance(working.index, pd.RangeIndex):
        working = working.reset_index()
    renamed: dict[str, str] = {}
    for column in working.columns:
        lowered = str(column).strip().lower().replace(" ", "_")
        renamed[column] = {
            "date": "timestamp",
            "datetime": "timestamp",
            "index": "timestamp",
            "adj_close": "adj_close",
            "adjclose": "adj_close",
            "close": "close",
            "open": "open",
            "high": "high",
            "low": "low",
            "volume": "volume",
        }.get(lowered, lowered)
    working = working.rename(columns=renamed)
    if "timestamp" not in working.columns:
        raise DataSourceError(f"provider frame has no date column; columns were {list(frame.columns)}")

    result = pd.DataFrame()
    result["symbol"] = [symbol.upper()] * len(working)
    result["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(working.get(column), errors="coerce")
    result["adj_close"] = pd.to_numeric(working["adj_close"], errors="coerce") if "adj_close" in working else pd.NA
    result["source"] = source_label
    result = result[MARKET_COLUMNS].sort_values("timestamp", kind="stable").reset_index(drop=True)
    return result


def fetch_market_data(
    symbol: str,
    start: date,
    end: date,
    *,
    interval: str = "1d",
    root: Path | str | None = None,
    source_label: str = DEFAULT_SOURCE_LABEL,
    downloader: Downloader | None = None,
    force: bool = False,
) -> Path:
    """Fetch (or reuse) the raw market CSV and refresh its manifest entry."""
    target = market_file_path(symbol, interval, start, end, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        logger.info("reusing cached market data: %s", relative_to_root(target, root))
        return target

    fetch = downloader or yfinance_downloader
    frame = normalise_download(fetch(symbol, start, end, interval), symbol=symbol, source_label=source_label)
    frame.to_csv(target, index=False, lineterminator="\n")
    logger.info("wrote %s rows to %s", len(frame), relative_to_root(target, root))

    coverage = frame["timestamp"].dropna()
    write_manifest(
        root,
        key=f"market:{symbol.upper()}:{interval}",
        entry=make_dataset_entry(
            key=f"market:{symbol.upper()}:{interval}",
            name=f"{symbol.upper()} {interval} OHLCV",
            file_path=target,
            root=root,
            source=source_label,
            url=f"https://finance.yahoo.com/quote/{symbol.upper()}",
            rows=int(len(frame)),
            columns=MARKET_COLUMNS,
            date_coverage=(coverage.min().date().isoformat(), coverage.max().date().isoformat()) if len(coverage) else None,
            notes=[
                "Bars are exchange sessions; the timestamp is a UTC date stamp, not an intraday instant.",
                "Requested window (inclusive of provider availability): "
                f"{start.isoformat()}..{end.isoformat()}.",
                "Adjacent sessions are not gap-filled; holidays simply have no row.",
            ],
        ),
    )
    return target


def load_market_data(path: Path | str) -> pd.DataFrame:
    """Load a cached market CSV exactly as stored (no cleaning, no imputation)."""
    target = Path(path)
    if not target.exists():
        raise DataSourceError(f"market data file not found: {target}")
    frame = pd.read_csv(target)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


__all__ = [
    "DEFAULT_SOURCE_LABEL",
    "MARKET_COLUMNS",
    "Downloader",
    "fetch_market_data",
    "load_market_data",
    "market_file_path",
    "market_filename",
    "normalise_download",
    "yfinance_downloader",
]
