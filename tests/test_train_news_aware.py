"""Phase 4 comparison: fair share of the split, artifact contract, and determinism."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.config import Config, load_config
from stock_project.common.errors import DataNotFetchedError
from stock_project.data.inspect_data import market_processed_path
from stock_project.features.news_features import news_feature_columns
from stock_project.models.train_baseline import run_baseline
from stock_project.models.train_news_aware import (
    PREDICTIONS_FILENAME,
    REPORT_FILENAME,
    main,
    news_features_processed_path,
    run_comparison,
)
from stock_project.nlp.classify_news import classified_path
from stock_project.nlp.taxonomy import EVENT_CATEGORIES

PREDICTION_COLUMNS = [
    "session_date",
    "decision_time_utc",
    "next_session_return",
    "target",
    "probability_up_market_only",
    "prediction_market_only",
    "probability_up_market_news",
    "prediction_market_news",
    "correct_market_only",
    "correct_market_news",
]


def prepare_inputs(root: Path, config: Config, synthetic_market, rows: int = 220) -> tuple[Path, Path]:
    """Write a synthetic market table and a classified-news table where the pipeline expects them."""
    market_target = market_processed_path(config, root=root)
    market_target.parent.mkdir(parents=True, exist_ok=True)
    synthetic_market(rows, seed=11).to_csv(market_target, index=False, lineterminator="\n")

    days = pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="UTC")
    labels = ["positive", "negative", "neutral"]
    scores = [0.5, -0.5, 0.0]
    news = pd.DataFrame(
        {
            "article_id": [f"article-{index:04d}" for index in range(len(days))],
            "published_at": (days - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "available_from": days.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_category": [EVENT_CATEGORIES[index % len(EVENT_CATEGORIES)] for index in range(len(days))],
            "sentiment_label": [labels[index % 3] for index in range(len(days))],
            "sentiment_score": [scores[index % 3] for index in range(len(days))],
        }
    )
    news_target = classified_path(config, root=root)
    news_target.parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(news_target, index=False, lineterminator="\n")
    return market_target, news_target


def test_run_writes_artifacts_and_every_check_passes(tmp_path: Path, config_path: Path, synthetic_market):
    config = load_config(config_path)
    prepare_inputs(tmp_path, config, synthetic_market)

    report = run_comparison(config, root=tmp_path)

    report_file = tmp_path / "reports" / "experiment_results" / REPORT_FILENAME
    on_disk = json.loads(report_file.read_text(encoding="utf-8"))
    assert on_disk["phase"] == 4
    assert on_disk["generated_at_utc"].endswith("Z")
    assert not [check for check in on_disk["checks"] if not check["passed"]], on_disk["checks"]
    assert len(on_disk["checks"]) == 10
    assert on_disk["next_step"].startswith("Phase 5")

    # Both models are scored on the same held-out rows.
    periods = on_disk["splits"]["periods"]
    assert periods["train"]["last_session"] < periods["validation"]["first_session"]
    assert periods["validation"]["last_session"] < periods["test"]["first_session"]
    for name in ("market_only", "market_plus_news"):
        assert on_disk["models"][name]["test"]["rows"] == periods["test"]["rows"]
        assert on_disk["models"][name]["fit_rows"] == periods["train"]["rows"] + periods["validation"]["rows"]

    # Feature sets are disjoint and complete.
    market_columns = on_disk["features"]["market_columns"]
    news_columns = on_disk["features"]["news_columns"]
    assert news_columns == news_feature_columns()
    assert not set(market_columns) & set(news_columns)
    assert on_disk["models"]["market_only"]["standardised_coefficients"].keys() == set(market_columns)
    assert on_disk["models"]["market_plus_news"]["standardised_coefficients"].keys() == set(market_columns + news_columns)

    comparison = on_disk["comparison"]
    assert comparison["news_minus_market_only"] == pytest.approx(
        comparison["market_plus_news"] - comparison["market_only"]
    )
    assert comparison["news_beats_market_only_on_test"] == (
        comparison["market_plus_news"] > comparison["market_only"]
    )

    features_file = news_features_processed_path(config, root=tmp_path)
    predictions_file = tmp_path / "reports" / "experiment_results" / PREDICTIONS_FILENAME
    assert features_file.exists() and predictions_file.exists()

    features = pd.read_csv(features_file)
    assert len(features) == on_disk["target_balance"]["rows"]
    for column in market_columns + news_columns:
        assert column in features.columns
    assert features["no_news"].isin([0, 1]).all()

    predictions = pd.read_csv(predictions_file)
    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == periods["test"]["rows"]
    assert predictions["session_date"].is_monotonic_increasing
    assert predictions["target"].equals(predictions["target"].astype(int))
    assert (predictions["prediction_market_only"].isin([0, 1])).all()
    assert (predictions["prediction_market_news"].isin([0, 1])).all()
    assert report["models"]["market_plus_news"]["test"]["rows"] == len(predictions)

    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert "processed:features_news:SPY:1d" in manifest["datasets"]


def test_market_only_metrics_match_the_phase_three_baseline(tmp_path: Path, config_path: Path, synthetic_market):
    """Fair comparison: adding news must not change the market-only model's inputs, split, or result."""
    config = load_config(config_path)
    prepare_inputs(tmp_path, config, synthetic_market)

    phase3 = run_baseline(config, root=tmp_path, write=False)
    phase4 = run_comparison(config, root=tmp_path, write=False)

    assert phase4["splits"]["ranges"] == phase3["splits"]["ranges"]
    assert phase4["models"]["market_only"]["selected_c"] == phase3["models"]["market_only"]["selected_c"]
    assert phase4["models"]["market_only"]["test"] == phase3["models"]["market_only"]["test"]


def test_run_is_deterministic(tmp_path: Path, config_path: Path, synthetic_market):
    config = load_config(config_path)
    prepare_inputs(tmp_path, config, synthetic_market)
    first = run_comparison(config, root=tmp_path, write=False)
    second = run_comparison(config, root=tmp_path, write=False)
    assert first["models"] == second["models"]
    assert first["comparison"] == second["comparison"]
    assert first["splits"] == second["splits"]


def test_missing_news_table_points_at_phase_two(tmp_path: Path, config_path: Path, synthetic_market):
    config = load_config(config_path)
    target = market_processed_path(config, root=tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    synthetic_market(220, seed=11).to_csv(target, index=False, lineterminator="\n")

    with pytest.raises(DataNotFetchedError, match="classify_news"):
        run_comparison(config, root=tmp_path)


def test_cli_exits_zero_and_prints_the_comparison(tmp_path: Path, config_path: Path, synthetic_market, capsys):
    config = load_config(config_path)
    prepare_inputs(tmp_path, config, synthetic_market)

    exit_code = main(["--config", str(config_path), "--root", str(tmp_path)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "All 10 Phase 4 checks passed." in captured
    assert "naive    :" in captured and "market   :" in captured and "+news    :" in captured
    assert "delta    :" in captured
    assert (tmp_path / "reports" / "experiment_results" / REPORT_FILENAME).exists()
