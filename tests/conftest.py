"""Shared test fixtures.

Every test in this suite runs **offline and deterministically**: no test downloads
market or news data, none depends on the current time, and all fixtures are tiny
hand-checkable frames whose expected numbers are asserted exactly.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MARKET_BARS_FIXTURE = FIXTURES_DIR / "market_bars.csv"
NEWS_SUBSET_FIXTURE = FIXTURES_DIR / "news_subset.csv"
NEWS_SOURCE_FIXTURE = FIXTURES_DIR / "news_source.jsonl"

#: Deliberately small, valid configuration used by most tests.
BASE_CONFIG: dict[str, Any] = {
    "project": {"name": "test-project", "random_seed": 7},
    "data": {
        "market": {
            "symbol": "SPY",
            "start": "2020-01-01",
            "end": "2020-01-10",
            "interval": "1d",
            "timezone": "UTC",
            "source": "fixture provider",
        },
        "news": {
            "dataset_id": "example/news",
            "revision": "deadbeef",
            "source_file": "data.json",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "dataset_url": "https://example.com/datasets/news",
            "source_category_filter": "POLITICS",
            "start": "2020-01-01",
            "end": "2020-01-06",
            "max_articles": 100,
            "timestamp_precision": "day",
        },
    },
    "nlp": {
        "text_scope": "headline",
        "taxonomy_version": "test-1",
        "sentiment_mode": "lexicon",
        "review_sample_size": 5,
        "review_sample_seed": 3,
    },
    "prediction": {"cutoff_time": "00:00", "target": "next_session_return_positive"},
    "trading": {"initial_capital": 100000.0, "transaction_cost_bps": 10.0, "position_fraction": 1.0},
}


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def dump_config(path: Path, mapping: dict[str, Any]) -> Path:
    """Write an arbitrary config mapping, including deliberately invalid ones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(mapping, sort_keys=False))
    return path


def write_config(path: Path, **overrides: Any) -> Path:
    """Write a valid config file, optionally overriding nested keys."""
    return dump_config(path, deep_merge(BASE_CONFIG, overrides))


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path / "configs" / "config.yaml")


@pytest.fixture
def config_writer():
    """Return :func:`write_config` so tests can build valid or invalid variants."""
    return write_config


@pytest.fixture
def config_dumper():
    """Return :func:`dump_config` for hand-built (possibly invalid) mappings."""
    return dump_config


@pytest.fixture
def base_config() -> dict[str, Any]:
    """A fresh copy of the valid test configuration, safe to mutate."""
    return copy.deepcopy(BASE_CONFIG)


@pytest.fixture
def market_window() -> tuple[date, date]:
    return date(2020, 1, 1), date(2020, 1, 10)


@pytest.fixture
def market_raw() -> pd.DataFrame:
    """Synthetic market rows: out of order, one duplicate pair, one unparsable timestamp."""
    return pd.read_csv(MARKET_BARS_FIXTURE, dtype=str)


@pytest.fixture
def market_provider_frame() -> pd.DataFrame:
    """A provider-shaped frame (DatetimeIndex plus provider column names)."""
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    return pd.DataFrame(
        {
            "Open": [98.0, 99.0, 100.0],
            "High": [100.0, 101.0, 102.0],
            "Low": [97.0, 98.0, 99.0],
            "Close": [99.0, 100.0, 101.0],
            "Adj Close": [99.0, 100.0, 101.0],
            "Volume": [900, 950, 1000],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=index,
    )


@pytest.fixture
def news_subset_raw() -> pd.DataFrame:
    """Synthetic news rows as written by the interim subset step."""
    return pd.read_csv(NEWS_SUBSET_FIXTURE, dtype=str)


#: Synthetic headlines, one per taxonomy category plus flagged evidence. Not real news:
#: they exist so the Phase 2 expectations are exact and hand-checkable.
CLASSIFIABLE_HEADLINES: tuple[tuple[str, str], ...] = (
    ("conflict", "Missiles strike port as troops advance"),
    ("sanctions", "EU imposes new sanctions on banks"),
    ("tariffs", "U.S. announces tariffs on steel imports"),
    ("pipeline", "Pipeline outage causes gas shortage in Europe"),
    ("summit", "Leaders hold summit talks on security"),
    ("council", "City council debates parking rules"),
    ("threat", "Trump threatens tariffs on Chinese goods"),
    ("denial", "No new sanctions expected, official says"),
)


@pytest.fixture
def news_raw_classifiable() -> pd.DataFrame:
    """Synthetic source-layout news table covering every event category."""
    return pd.DataFrame(
        [
            {
                "link": f"https://www.example.com/{slug}",
                "headline": headline,
                "category": "POLITICS",
                "short_description": f"Description for {slug}.",
                "authors": "Author",
                "date": f"2020-01-{index + 1:02d}",
            }
            for index, (slug, headline) in enumerate(CLASSIFIABLE_HEADLINES)
        ]
    )
