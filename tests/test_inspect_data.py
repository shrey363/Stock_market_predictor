"""End-to-end Phase 1 inspection on cached fixtures.

This is the artifact contract test: it runs the same code path the CLI runs and
checks the JSON report, the cleaned tables, and the manifest the next phase reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.config import Config, load_config
from stock_project.common.errors import DataNotFetchedError
from stock_project.data.inspect_data import (
    REPORT_FILENAME,
    inspect_datasets,
    main,
    market_processed_path,
    news_processed_path,
)
from stock_project.data.market_data import market_file_path
from stock_project.data.news_data import subset_path
from stock_project.preprocessing.clean_market_data import MARKET_COLUMNS
from stock_project.preprocessing.clean_news_data import NEWS_COLUMNS

FIXTURES = Path(__file__).parent / "fixtures"


def prepare_cached_project(root: Path, config_path: Path) -> Config:
    """Copy the synthetic fixtures into the locations the pipeline expects."""
    config = load_config(config_path)
    market_target = market_file_path(
        config.market.symbol, config.market.interval, config.market.start, config.market.end, root=root
    )
    market_target.parent.mkdir(parents=True, exist_ok=True)
    market_target.write_bytes((FIXTURES / "market_bars.csv").read_bytes())

    news_target = subset_path(config.news.source_category_filter, config.news.start, config.news.end, root=root)
    news_target.parent.mkdir(parents=True, exist_ok=True)
    news_target.write_bytes((FIXTURES / "news_subset.csv").read_bytes())
    return config


def test_report_is_written_and_every_check_passes(tmp_path: Path, config_path: Path):
    config = prepare_cached_project(tmp_path, config_path)
    report = inspect_datasets(config, root=tmp_path)

    report_file = tmp_path / "reports" / "experiment_results" / REPORT_FILENAME
    assert report_file.exists()
    on_disk = json.loads(report_file.read_text(encoding="utf-8"))

    assert on_disk["phase"] == 1
    assert on_disk["generated_at_utc"].endswith("Z")
    failed = [check for check in on_disk["checks"] if not check["passed"]]
    assert not failed, failed
    assert len(on_disk["checks"]) == 10
    assert on_disk["limitations"]
    assert on_disk["unassigned_columns"] == [
        "event_category",
        "sentiment_label",
        "sentiment_score",
        "affected_sector",
    ]
    assert on_disk["next_step"].startswith("Phase 2")
    assert on_disk["datasets"]["market"]["sessions"] == report["datasets"]["market"]["sessions"]


def test_report_numbers_match_the_fixtures(tmp_path: Path, config_path: Path):
    config = prepare_cached_project(tmp_path, config_path)
    report = inspect_datasets(config, root=tmp_path)

    market = report["datasets"]["market"]
    news = report["datasets"]["news"]
    assert market["sessions"] == 3
    assert market["first_session"] == "2020-01-02"
    assert market["last_session"] == "2020-01-06"
    assert market["quality"]["duplicates_removed"] == 2
    assert market["quality"]["imputation_applied"] == []
    assert news["raw_rows"] == 6
    assert news["articles"] == 2
    assert news["first_published"] == "2020-01-02"
    assert news["last_published"] == "2020-01-05"
    assert news["quality"]["duplicates_removed"] == 2
    assert news["quality"]["rows_written"] == 2
    assert report["summary"]["news_publishers"] == {"example.com": 2}
    assert "available_from" in report["summary"]["news_columns"]


def test_artifact_paths_are_project_relative(tmp_path: Path, config_path: Path):
    config = prepare_cached_project(tmp_path, config_path)
    report = inspect_datasets(config, root=tmp_path)
    assert report["datasets"]["market"]["raw_file"] == "data/raw/market/SPY_1d_20200101_20200110.csv"
    assert report["datasets"]["market"]["processed_file"].startswith("data/processed/")
    assert report["datasets"]["news"]["processed_file"].startswith("data/processed/")
    assert str(tmp_path) not in report["datasets"]["market"]["raw_file"]


def test_processed_tables_and_manifest_are_written(tmp_path: Path, config_path: Path):
    config = prepare_cached_project(tmp_path, config_path)
    inspect_datasets(config, root=tmp_path)

    market_out = market_processed_path(config, root=tmp_path)
    news_out = news_processed_path(config, root=tmp_path)
    assert market_out.exists() and news_out.exists()

    market = pd.read_csv(market_out)
    news = pd.read_csv(news_out)
    assert list(market.columns) == MARKET_COLUMNS
    assert list(news.columns) == NEWS_COLUMNS
    assert len(market) == 3 and len(news) == 2

    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert "processed:market:SPY:1d" in manifest["datasets"]
    assert "processed:news:example/news" in manifest["datasets"]
    assert manifest["datasets"]["processed:news:example/news"]["license"] == "CC BY 4.0"


def test_missing_cached_data_raises_with_the_next_command(tmp_path: Path, config_path: Path):
    config = load_config(config_path)
    with pytest.raises(DataNotFetchedError, match="fetch_data"):
        inspect_datasets(config, root=tmp_path)


def test_cli_exits_zero_and_prints_the_summary(tmp_path: Path, config_path: Path, capsys):
    prepare_cached_project(tmp_path, config_path)
    exit_code = main(["--config", str(config_path), "--root", str(tmp_path)])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "All 10 Phase 1 checks passed." in captured
    assert (tmp_path / "reports" / "experiment_results" / REPORT_FILENAME).exists()
