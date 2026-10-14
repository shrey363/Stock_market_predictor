"""Configuration loading and validation. Offline: touches no data source."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from stock_project.common.config import (
    ConfigError,
    default_config_path,
    load_config,
    parse_iso_date,
)
from stock_project.common.paths import PROJECT_ROOT

EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "config.example.yaml"


def test_committed_example_config_is_valid():
    config = load_config(EXAMPLE_CONFIG)
    assert config.market.symbol == "SPY"
    assert config.market.interval == "1d"
    assert config.market.timezone == "UTC"
    assert config.market.start == date(2018, 1, 1)
    assert config.market.end == date(2022, 12, 31)
    assert config.news.source_category_filter == "POLITICS"
    assert config.news.license == "CC BY 4.0"
    assert config.news.max_articles > 0
    assert config.prediction.cutoff_time == "00:00"
    assert config.trading.initial_capital == 100000.0
    assert config.trading.transaction_cost_bps == 10.0


def test_config_snapshot_is_json_serialisable():
    snapshot = json.loads(json.dumps(load_config(EXAMPLE_CONFIG).to_dict()))
    assert snapshot["data"]["market"]["symbol"] == "SPY"
    assert snapshot["data"]["news"]["revision"]
    assert snapshot["project"]["random_seed"] == 42


def test_valid_test_config_loads(config_path: Path):
    config = load_config(config_path)
    assert config.market.start == date(2020, 1, 1)
    assert config.news.dataset_id == "example/news"


def test_missing_section_names_the_section(config_dumper, base_config: dict, tmp_path: Path):
    del base_config["trading"]
    path = config_dumper(tmp_path / "no-trading.yaml", base_config)
    with pytest.raises(ConfigError, match="trading"):
        load_config(path)


def test_start_after_end_is_rejected(config_writer, tmp_path: Path):
    path = config_writer(tmp_path / "bad-window.yaml", data={"market": {"start": "2020-02-01", "end": "2020-01-01"}})
    with pytest.raises(ConfigError, match="must be earlier than"):
        load_config(path)


def test_non_utc_timezone_is_rejected(config_writer, tmp_path: Path):
    path = config_writer(tmp_path / "bad-tz.yaml", data={"market": {"timezone": "US/Eastern"}})
    with pytest.raises(ConfigError, match="UTC"):
        load_config(path)


def test_ambiguous_date_format_is_rejected(config_writer, tmp_path: Path):
    path = config_writer(tmp_path / "ambiguous-date.yaml", data={"market": {"start": "01/02/2020"}})
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        load_config(path)


def test_missing_key_is_named(config_dumper, base_config: dict, tmp_path: Path):
    del base_config["project"]["random_seed"]
    path = config_dumper(tmp_path / "no-seed.yaml", base_config)
    with pytest.raises(ConfigError, match="random_seed"):
        load_config(path)


def test_invalid_cutoff_time_is_rejected(config_writer, tmp_path: Path):
    path = config_writer(tmp_path / "bad-cutoff.yaml", prediction={"cutoff_time": "9:5"})
    with pytest.raises(ConfigError, match="HH:MM"):
        load_config(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"trading": {"initial_capital": 0}},
        {"trading": {"transaction_cost_bps": -1.0}},
        {"trading": {"position_fraction": 0}},
        {"trading": {"position_fraction": 1.5}},
        {"data": {"news": {"max_articles": 0}}},
    ],
)
def test_out_of_range_numbers_are_rejected(config_writer, tmp_path: Path, overrides: dict):
    path = config_writer(tmp_path / "out-of-range.yaml", **overrides)
    with pytest.raises(ConfigError):
        load_config(path)


def test_symbol_is_normalised_to_uppercase(config_writer, tmp_path: Path):
    path = config_writer(tmp_path / "lowercase-symbol.yaml", data={"market": {"symbol": " spy"}})
    assert load_config(path).market.symbol == "SPY"


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does-not-exist.yaml")


def test_default_config_path_prefers_local_config_then_example(tmp_path: Path):
    fallback = default_config_path(tmp_path)
    assert fallback.name == "config.example.yaml"
    (tmp_path / "configs").mkdir()
    preferred = tmp_path / "configs" / "config.yaml"
    preferred.write_text("project:\n", encoding="utf-8")
    assert default_config_path(tmp_path) == preferred


def test_parse_iso_date_accepts_date_objects():
    assert parse_iso_date(date(2021, 5, 4), where="test") == date(2021, 5, 4)
