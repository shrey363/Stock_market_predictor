"""Phase 6 RL protocol, risk metrics, artifact contract and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.config import Config, load_config
from stock_project.common.errors import DataNotFetchedError
from stock_project.models.train_news_aware import run_comparison
from stock_project.nlp.classify_news import classified_path
from stock_project.nlp.taxonomy import EVENT_CATEGORIES
from stock_project.data.inspect_data import market_processed_path
from stock_project.trading.simulator import SimulationSettings, run_simulation
from stock_project.trading.train_rl import (
    REPORT_FILENAME,
    VALUES_FILENAME,
    main,
    run_rl,
    sharpe_ratio,
)
def prepare_phase6(root: Path, config: Config, synthetic_market, rows: int = 220) -> None:
    """Write deterministic Phase 1–4 inputs without touching the network."""
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
    run_comparison(config, root=root)


def test_sharpe_ratio_convention_is_hand_checkable():
    settings = SimulationSettings(1000.0, 1.0, 0.0)
    prices = pd.DataFrame(
        {
            "session_date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "open": [10.0, 11.0, 9.9],
            "close": [11.0, 9.9, 11.88],
        }
    )
    equity = run_simulation(prices, [1, 0, 0], settings)["equity"]
    # Values 1100, 990, 1188 produce returns +10%, -10%, +20%.
    expected = ((0.1 - 0.1 + 0.2) / 3) / pd.Series([0.1, -0.1, 0.2]).std(ddof=1) * (252**0.5)
    assert sharpe_ratio(equity, settings) == pytest.approx(round(expected, 6))


def test_run_writes_artifacts_and_holds_test_out(
    tmp_path: Path, config_path: Path, synthetic_market, monkeypatch
):
    config = load_config(config_path)
    prepare_phase6(tmp_path, config, synthetic_market)

    fit_ranges: list[tuple[str, str]] = []

    class FixedModel:
        def predict(self, observation, deterministic=True):
            return 0, None

    def fake_fit(observations, sessions, settings, candidate, *, timesteps, seed):
        fit_ranges.append((str(sessions["session_date"].iloc[0]), str(sessions["session_date"].iloc[-1])))
        return FixedModel()

    monkeypatch.setattr("stock_project.trading.train_rl._fit_agent", fake_fit)
    report = run_rl(config, root=tmp_path)

    on_disk = json.loads(
        (tmp_path / "reports" / "experiment_results" / REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert on_disk["phase"] == 6
    assert on_disk["generated_at_utc"].endswith("Z")
    assert len(on_disk["checks"]) == 15
    assert not [check for check in on_disk["checks"] if not check["passed"]], on_disk["checks"]
    assert on_disk["next_step"].startswith("Phase 7")
    assert set(on_disk["test"]["benchmarks"]) == {"buy_and_hold", "momentum_5", "cash"}
    assert len(on_disk["test"]["agent"]["per_seed"]) == len(config.rl.seeds)

    periods = on_disk["splits"]["periods"]
    assert all(last <= periods["validation"]["last_session"] for _, last in fit_ranges)
    assert all(last < periods["test"]["first_session"] for _, last in fit_ranges)
    assert periods["train"]["last_session"] < periods["validation"]["first_session"]
    assert periods["validation"]["last_session"] < periods["test"]["first_session"]

    values = pd.read_csv(tmp_path / "reports" / "experiment_results" / VALUES_FILENAME)
    expected_strategies = {"ppo_seed_0", "ppo_seed_1", "buy_and_hold", "momentum_5", "cash"}
    assert set(values["strategy"]) == expected_strategies
    assert len(values) == periods["test"]["rows"] * len(expected_strategies)
    assert report["test"]["agent"]["aggregate"]["mean_total_return"] == 0.0


def test_missing_feature_table_points_at_phase_four(tmp_path: Path, config_path: Path):
    config = load_config(config_path)
    with pytest.raises(DataNotFetchedError, match="train_news_aware"):
        run_rl(config, root=tmp_path)


def test_cli_exits_zero_and_prints_results(
    tmp_path: Path, config_path: Path, synthetic_market, monkeypatch, capsys
):
    config = load_config(config_path)
    prepare_phase6(tmp_path, config, synthetic_market)

    class FixedModel:
        def predict(self, observation, deterministic=True):
            return 0, None

    monkeypatch.setattr(
        "stock_project.trading.train_rl._fit_agent",
        lambda *args, **kwargs: FixedModel(),
    )
    exit_code = main(["--config", str(config_path), "--root", str(tmp_path)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "All 15 Phase 6 checks passed." in captured
    assert "ppo seed" in captured and "buy_and_hold" in captured
