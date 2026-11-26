"""Phase 5 simulator: hand-checkable cash/position arithmetic, costs, and determinism.

Every expected number below is computed by hand from a tiny price series. The
benchmark tests assert that action generation reads no future price.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.config import load_config
from stock_project.common.errors import DataNotFetchedError, DataSourceError
from stock_project.data.inspect_data import market_processed_path
from stock_project.trading.simulator import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    EQUITY_FILENAME,
    REPORT_FILENAME,
    SimulationSettings,
    buy_and_hold_actions,
    main,
    momentum_actions,
    run_benchmarks,
    run_simulation,
)


def frame(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": [row[0] for row in rows],
            "open": [row[1] for row in rows],
            "close": [row[2] for row in rows],
        }
    )


def settings(**overrides) -> SimulationSettings:
    values = {"initial_capital": 1000.0, "position_fraction": 1.0, "transaction_cost_bps": 0.0}
    values.update(overrides)
    return SimulationSettings(**values)


def test_buy_then_mark_to_market():
    prices = frame([("2020-01-02", 10.0, 11.0)])
    result = run_simulation(prices, [ACTION_BUY], settings())
    row = result["equity"].iloc[0]
    assert row["executed"]
    assert row["trade_shares"] == 100
    assert row["cash"] == pytest.approx(0.0)
    assert row["position_value"] == pytest.approx(1100.0)
    assert row["portfolio_value"] == pytest.approx(1100.0)
    assert result["summary"]["total_return"] == pytest.approx(0.10)


def test_transaction_cost_is_charged_on_notional():
    prices = frame([("2020-01-02", 10.0, 11.0)])
    result = run_simulation(prices, [ACTION_BUY], settings(transaction_cost_bps=100.0))  # 1%
    row = result["equity"].iloc[0]
    # 100 shares would cost 1000 + 10; only 99 are affordable: 990 notional + 9.9 cost.
    assert row["trade_shares"] == 99
    assert row["trade_notional"] == pytest.approx(990.0)
    assert row["transaction_cost"] == pytest.approx(9.9)
    assert row["cash"] == pytest.approx(0.1)
    assert row["portfolio_value"] == pytest.approx(1089.1)
    assert result["summary"]["total_transaction_costs"] == pytest.approx(9.9)


def test_buy_while_long_does_nothing():
    prices = frame([("2020-01-02", 10.0, 11.0), ("2020-01-03", 12.0, 13.0)])
    result = run_simulation(prices, [ACTION_BUY, ACTION_BUY], settings())
    second = result["equity"].iloc[1]
    assert not second["executed"]
    assert second["reason"] == "already_long"
    assert second["shares_after"] == 100
    assert second["portfolio_value"] == pytest.approx(1300.0)


def test_sell_closes_the_entire_position():
    prices = frame([("2020-01-02", 10.0, 11.0), ("2020-01-03", 12.0, 13.0), ("2020-01-06", 5.0, 5.0)])
    result = run_simulation(prices, [ACTION_BUY, ACTION_SELL, ACTION_HOLD], settings())
    sold = result["equity"].iloc[1]
    assert sold["executed"]
    assert sold["reason"] == "sold"
    assert sold["trade_shares"] == -100
    assert sold["cash"] == pytest.approx(1200.0)
    assert sold["shares_after"] == 0
    assert result["equity"].iloc[2]["portfolio_value"] == pytest.approx(1200.0)
    assert result["summary"]["number_of_trades"] == 2


def test_sell_when_flat_does_nothing():
    prices = frame([("2020-01-02", 10.0, 11.0)])
    result = run_simulation(prices, [ACTION_SELL], settings())
    row = result["equity"].iloc[0]
    assert not row["executed"]
    assert row["reason"] == "no_position"
    assert row["cash"] == pytest.approx(1000.0)


def test_unaffordable_purchase_never_overspends():
    prices = frame([("2020-01-02", 10.0, 10.0)])
    result = run_simulation(prices, [ACTION_BUY], settings(initial_capital=5.0))
    row = result["equity"].iloc[0]
    assert not row["executed"]
    assert row["reason"] == "no_affordable_shares"
    assert row["cash"] == pytest.approx(5.0)
    assert row["shares_after"] == 0


def test_position_fraction_limits_the_purchase():
    prices = frame([("2020-01-02", 10.0, 10.0)])
    result = run_simulation(prices, [ACTION_BUY], settings(position_fraction=0.5))
    row = result["equity"].iloc[0]
    assert row["trade_shares"] == 50
    assert row["cash"] == pytest.approx(500.0)
    assert row["portfolio_value"] == pytest.approx(1000.0)


def test_probability_of_negative_cash_is_zero_across_a_mixed_sequence():
    prices = frame(
        [
            ("2020-01-02", 10.0, 12.0),
            ("2020-01-03", 20.0, 5.0),
            ("2020-01-06", 7.0, 8.0),
            ("2020-01-07", 9.0, 1.0),
        ]
    )
    actions = [ACTION_BUY, ACTION_SELL, ACTION_BUY, ACTION_HOLD]
    result = run_simulation(prices, actions, settings(transaction_cost_bps=25.0))
    equity = result["equity"]
    assert (equity["cash"] >= -1e-9).all()
    assert (equity["shares_after"] >= 0).all()


def test_unknown_action_is_rejected():
    prices = frame([("2020-01-02", 10.0, 11.0)])
    with pytest.raises(ValueError, match="unknown action"):
        run_simulation(prices, [9], settings())


def test_action_count_must_match_sessions():
    prices = frame([("2020-01-02", 10.0, 11.0), ("2020-01-03", 11.0, 12.0)])
    with pytest.raises(ValueError, match="expected 2 actions"):
        run_simulation(prices, [ACTION_BUY], settings())


def test_invalid_prices_are_rejected():
    with pytest.raises(DataSourceError, match="missing values"):
        run_simulation(frame([("2020-01-02", 10.0, float("nan"))]), [ACTION_HOLD], settings())
    with pytest.raises(DataSourceError, match="must be positive"):
        run_simulation(frame([("2020-01-02", 0.0, 11.0)]), [ACTION_HOLD], settings())


def test_sessions_must_be_chronological():
    prices = frame([("2020-01-03", 10.0, 11.0), ("2020-01-02", 10.0, 11.0)])
    with pytest.raises(DataSourceError, match="ascending chronological order"):
        run_simulation(prices, [ACTION_HOLD, ACTION_HOLD], settings())


def test_replay_is_deterministic():
    prices = frame([("2020-01-02", 10.0, 11.0), ("2020-01-03", 12.0, 13.0)])
    first = run_simulation(prices, [ACTION_BUY, ACTION_HOLD], settings(transaction_cost_bps=10.0))
    second = run_simulation(prices, [ACTION_BUY, ACTION_HOLD], settings(transaction_cost_bps=10.0))
    pd.testing.assert_frame_equal(first["equity"], second["equity"])
    assert first["summary"] == second["summary"]


def test_benchmark_action_generators():
    assert buy_and_hold_actions(3) == [ACTION_BUY, ACTION_HOLD, ACTION_HOLD]

    prices = frame(
        [
            ("2020-01-02", 10.0, 10.0),
            ("2020-01-03", 11.0, 11.0),
            ("2020-01-06", 12.0, 12.0),
            ("2020-01-07", 13.0, 13.0),
            ("2020-01-08", 14.0, 14.0),
            ("2020-01-09", 15.0, 15.0),
            ("2020-01-10", 9.0, 9.0),
        ]
    )
    # window=2: mean of the two closes before the decision session.
    assert momentum_actions(prices, window=2) == [
        ACTION_HOLD,  # no previous close yet
        ACTION_SELL,  # close[0]=10 is not above mean(10)=10
        ACTION_BUY,  # close[1]=11 > mean(10,11)=10.5
        ACTION_BUY,  # close[2]=12 > mean(11,12)=11.5
        ACTION_BUY,  # close[3]=13 > mean(12,13)=12.5
        ACTION_BUY,  # close[4]=14 > mean(13,14)=13.5
        ACTION_BUY,  # close[5]=15 > mean(14,15)=14.5
    ]


def test_momentum_actions_do_not_read_the_session_they_trade():
    prices = frame([("2020-01-0%d" % day, 10.0, 10.0 + day) for day in range(2, 9)])
    baseline = momentum_actions(prices, window=3)
    assert baseline[4] == ACTION_BUY

    mutated = prices.copy()
    mutated.loc[4, "close"] = 1.0  # session 5's own close must not affect its action
    changed = momentum_actions(mutated, window=3)

    # Actions up to and including session 5 are unchanged; session 6 (which may
    # read session 5's close) is allowed to change.
    assert changed[:5] == baseline[:5]
    assert changed[5] != baseline[5]


def test_run_writes_artifacts_and_every_check_passes(tmp_path: Path, config_path: Path, synthetic_market):
    config = load_config(config_path)
    target = market_processed_path(config, root=tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    synthetic_market(220, seed=11).to_csv(target, index=False, lineterminator="\n")

    report = run_benchmarks(config, root=tmp_path)

    report_file = tmp_path / "reports" / "experiment_results" / REPORT_FILENAME
    on_disk = json.loads(report_file.read_text(encoding="utf-8"))
    assert on_disk["phase"] == 5
    assert on_disk["generated_at_utc"].endswith("Z")
    assert not [check for check in on_disk["checks"] if not check["passed"]], on_disk["checks"]
    assert len(on_disk["checks"]) == 13
    assert on_disk["next_step"].startswith("Phase 6")
    assert set(on_disk["strategies"]) == {"buy_and_hold", "momentum_5"}
    for name, summary in on_disk["strategies"].items():
        assert summary["sessions"] == 220
        assert summary["number_of_trades"] >= 1
        assert summary["final_value"] > 0
        assert 0.0 <= summary["max_drawdown"] <= 1.0
    assert on_disk["strategies"]["buy_and_hold"]["number_of_trades"] == 1

    equity_file = tmp_path / "reports" / "experiment_results" / EQUITY_FILENAME
    equity = pd.read_csv(equity_file)
    assert len(equity) == 2 * 220
    assert set(equity["strategy"]) == {"buy_and_hold", "momentum_5"}
    assert (equity["cash"] >= -1e-9).all()
    assert (equity["shares_after"] >= 0).all()


def test_missing_market_table_points_at_phase_one(tmp_path: Path, config_path: Path):
    config = load_config(config_path)
    with pytest.raises(DataNotFetchedError, match="inspect_data"):
        run_benchmarks(config, root=tmp_path)


def test_cli_exits_zero_and_prints_benchmarks(tmp_path: Path, config_path: Path, synthetic_market, capsys):
    config = load_config(config_path)
    target = market_processed_path(config, root=tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    synthetic_market(220, seed=11).to_csv(target, index=False, lineterminator="\n")

    exit_code = main(["--config", str(config_path), "--root", str(tmp_path)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "All 13 Phase 5 checks passed." in captured
    assert "buy_and_hold:" in captured and "momentum_5" in captured
    assert (tmp_path / "reports" / "experiment_results" / REPORT_FILENAME).exists()
