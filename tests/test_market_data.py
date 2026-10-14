"""Market ingestion: normalisation, caching, provenance, and offline guarantees.

The provider is always either a fake callable or absent, so this module never
performs a network request.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.errors import DataSourceError
from stock_project.common.manifest import read_manifest, sha256_file
from stock_project.data.market_data import (
    MARKET_COLUMNS,
    fetch_market_data,
    load_market_data,
    market_file_path,
    normalise_download,
    yfinance_downloader,
)

WINDOW = (date(2020, 1, 1), date(2020, 1, 10))


def test_fetch_writes_raw_file_and_manifest(tmp_path: Path, market_provider_frame: pd.DataFrame):
    calls: list[tuple] = []

    def fake_downloader(symbol, start, end, interval):
        calls.append((symbol, start, end, interval))
        return market_provider_frame

    path = fetch_market_data(
        "SPY", *WINDOW, root=tmp_path, source_label="fixture provider", downloader=fake_downloader
    )
    assert path == market_file_path("SPY", "1d", *WINDOW, root=tmp_path)
    assert calls == [("SPY", WINDOW[0], WINDOW[1], "1d")]

    frame = pd.read_csv(path)
    assert list(frame.columns) == MARKET_COLUMNS
    assert len(frame) == 3

    entry = read_manifest(tmp_path)["datasets"]["market:SPY:1d"]
    assert entry["source"] == "fixture provider"
    assert entry["rows"] == 3
    assert entry["date_coverage"] == ["2020-01-02", "2020-01-06"]
    assert entry["sha256"] == sha256_file(path)
    assert entry["file"] == "data/raw/market/SPY_1d_20200101_20200110.csv"
    assert entry["retrieved_at_utc"].endswith("Z")


def test_cached_file_is_reused_unless_forced(tmp_path: Path, market_provider_frame: pd.DataFrame):
    calls: list[str] = []

    def fake_downloader(symbol, start, end, interval):
        calls.append(symbol)
        return market_provider_frame

    first = fetch_market_data("SPY", *WINDOW, root=tmp_path, downloader=fake_downloader)
    second = fetch_market_data("SPY", *WINDOW, root=tmp_path, downloader=fake_downloader)
    assert first == second
    assert len(calls) == 1

    fetch_market_data("SPY", *WINDOW, root=tmp_path, downloader=fake_downloader, force=True)
    assert len(calls) == 2


def test_normalise_download_maps_columns_and_drops_provider_extras(market_provider_frame: pd.DataFrame):
    normalised = normalise_download(market_provider_frame, symbol="spy", source_label="fixture provider")
    assert list(normalised.columns) == MARKET_COLUMNS
    assert set(normalised["symbol"]) == {"SPY"}
    assert normalised["timestamp"].dt.tz is not None
    assert normalised["adj_close"].iloc[0] == pytest.approx(99.0)
    assert normalised["timestamp"].is_monotonic_increasing


def test_normalise_download_requires_a_date_column():
    frame = pd.DataFrame({"Open": [1.0], "Close": [1.0]})
    with pytest.raises(DataSourceError, match="no date column"):
        normalise_download(frame, symbol="SPY")


def test_normalise_download_rejects_empty_frames():
    with pytest.raises(DataSourceError, match="empty market frame"):
        normalise_download(pd.DataFrame(), symbol="SPY")


def test_missing_yfinance_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(DataSourceError, match="yfinance is not installed"):
        yfinance_downloader("SPY", *WINDOW, "1d")


def test_load_market_data_round_trips_utc_timestamps(tmp_path: Path, market_provider_frame: pd.DataFrame):
    path = fetch_market_data(
        "SPY", *WINDOW, root=tmp_path, downloader=lambda *args: market_provider_frame
    )
    loaded = load_market_data(path)
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    assert loaded["timestamp"].min() == pd.Timestamp("2020-01-02T00:00:00Z")
    assert loaded["close"].tolist() == [99.0, 100.0, 101.0]


def test_load_market_data_missing_file_raises(tmp_path: Path):
    with pytest.raises(DataSourceError, match="not found"):
        load_market_data(tmp_path / "absent.csv")
