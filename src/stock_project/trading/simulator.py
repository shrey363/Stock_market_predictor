"""Deterministic historical trading simulator (Phase 5).

Mechanics, stated explicitly because later phases depend on them
----------------------------------------------------------------
* **Decision and execution.** A session's action is decided pre-open and filled at
  that same session's **open** price. The portfolio is marked to market at that
  session's **close**. This is the only fill assumption; there is no slippage,
  partial fill, or liquidity model (recorded as a limitation).
* **Actions.** ``hold`` (0) does nothing, ``buy`` (1) opens a long position only
  when flat, ``sell`` (2) closes the entire position. Redundant actions (buy while
  long, sell while flat) execute nothing and are recorded with a reason.
* **Position limit.** Long-only, one position, no pyramiding and no shorting.
  A buy invests ``position_fraction`` of current equity in **whole shares**,
  capped by available cash including the transaction cost.
* **Costs.** ``transaction_cost_bps`` is charged on the traded notional
  (``bps / 10_000``), on both buys and sells. Cash can never go negative; if the
  requested size is unaffordable the trade is reduced, and if zero shares are
  affordable nothing is traded.
* **Determinism.** The engine is pure arithmetic over the supplied prices and
  actions: identical inputs always produce identical cash, shares and values
  (``docs/ARCHITECTURE.md`` §3, invariant 5).

Benchmarks in this phase are action generators, not part of the engine: buy-and-hold
and a 5-session price-momentum rule. Both use only information available before the
session they act in.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..common.config import Config, load_config
from ..common.errors import DataNotFetchedError, DataSourceError
from ..common.logging_utils import configure_cli_logging, get_logger
from ..common.paths import ensure_directories, relative_to_root, results_dir
from ..common.timeutils import utc_now_iso
from ..data.inspect_data import market_processed_path

logger = get_logger("stock_project.trading.simulator")

REPORT_FILENAME = "phase5_simulation_report.json"
EQUITY_FILENAME = "phase5_portfolio_values.csv"

ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_NAMES = {ACTION_HOLD: "hold", ACTION_BUY: "buy", ACTION_SELL: "sell"}

#: Columns a simulation frame must provide.
REQUIRED_SESSION_COLUMNS = ("session_date", "open", "close")

#: Momentum benchmark lookback, in sessions.
MOMENTUM_WINDOW = 5


@dataclass(frozen=True)
class SimulationSettings:
    """Capital, sizing and cost assumptions of one simulation run."""

    initial_capital: float
    position_fraction: float
    transaction_cost_bps: float

    def __post_init__(self) -> None:
        if not self.initial_capital > 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        if not 0 < self.position_fraction <= 1:
            raise ValueError(f"position_fraction must be in (0, 1], got {self.position_fraction}")
        if self.transaction_cost_bps < 0:
            raise ValueError(f"transaction_cost_bps must not be negative, got {self.transaction_cost_bps}")

    @property
    def cost_rate(self) -> float:
        return self.transaction_cost_bps / 10_000.0


def settings_from_config(config: Config) -> SimulationSettings:
    return SimulationSettings(
        initial_capital=config.trading.initial_capital,
        position_fraction=config.trading.position_fraction,
        transaction_cost_bps=config.trading.transaction_cost_bps,
    )


def _validate_price(value: Any, *, column: str, session: str) -> float:
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise DataSourceError(f"{column} for session {session!r} is not numeric: {value!r}") from exc
    if not math.isfinite(price) or price <= 0:
        raise DataSourceError(f"{column} for session {session!r} must be finite and positive, got {value!r}")
    return price


class PortfolioSimulator:
    """State machine for one portfolio: cash, shares, and cost accounting."""

    def __init__(self, settings: SimulationSettings) -> None:
        self.settings = settings
        self.cash = float(settings.initial_capital)
        self.shares = 0
        self.total_costs = 0.0

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def step(self, action: int, open_price: float, close_price: float) -> dict[str, object]:
        """Execute ``action`` at the open, then mark the portfolio at the close."""
        if action not in ACTION_NAMES:
            raise ValueError(f"unknown action {action!r}; expected one of {sorted(ACTION_NAMES)}")
        open_price = _validate_price(open_price, column="open", session="step")
        close_price = _validate_price(close_price, column="close", session="step")

        shares_before = self.shares
        notional = 0.0
        cost = 0.0
        executed = False
        reason = "hold"

        if action == ACTION_BUY:
            if self.shares > 0:
                reason = "already_long"
            else:
                equity = self.equity(open_price)
                target_shares = int(math.floor(self.settings.position_fraction * equity / open_price))
                max_shares = int(math.floor(self.cash / (open_price * (1.0 + self.settings.cost_rate))))
                quantity = max(0, min(target_shares, max_shares))
                if quantity == 0:
                    reason = "no_affordable_shares"
                else:
                    notional = quantity * open_price
                    cost = notional * self.settings.cost_rate
                    self.cash -= notional + cost
                    if self.cash < 0 and self.cash > -1e-9:  # floating-point dust only
                        self.cash = 0.0
                    self.shares += quantity
                    self.total_costs += cost
                    executed = True
                    reason = "bought"
        elif action == ACTION_SELL:
            if self.shares == 0:
                reason = "no_position"
            else:
                notional = self.shares * open_price
                cost = notional * self.settings.cost_rate
                self.cash += notional - cost
                self.shares = 0
                self.total_costs += cost
                executed = True
                reason = "sold"

        position_value = self.shares * close_price
        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "executed": executed,
            "reason": reason,
            "shares_before": shares_before,
            "shares_after": self.shares,
            "trade_shares": self.shares - shares_before,
            "price_open": open_price,
            "price_close": close_price,
            "trade_notional": notional,
            "transaction_cost": cost,
            "cash": self.cash,
            "position_value": position_value,
            "portfolio_value": self.cash + position_value,
        }


def _prepare_sessions(sessions: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    missing = [column for column in REQUIRED_SESSION_COLUMNS if column not in sessions.columns]
    if missing:
        raise DataSourceError(
            f"session frame is missing required column(s) {missing}; present columns: {list(sessions.columns)}"
        )
    if sessions.empty:
        raise DataSourceError("session frame has no rows to simulate")

    working = sessions.loc[:, list(REQUIRED_SESSION_COLUMNS)].copy()
    working["session_date"] = working["session_date"].astype("string")
    if working["session_date"].isna().any() or working["session_date"].duplicated().any():
        raise DataSourceError("session_date must be present and unique on every row")
    parsed_dates = pd.to_datetime(working["session_date"], errors="coerce", utc=True)
    if parsed_dates.isna().any() or not (parsed_dates.dt.strftime("%Y-%m-%d") == working["session_date"]).all():
        raise DataSourceError("session_date must be a YYYY-MM-DD date on every row")
    chronological = bool(working["session_date"].is_monotonic_increasing)
    if not chronological:
        raise DataSourceError(
            "sessions must arrive in ascending chronological order; the simulator never reorders rows "
            "because that would silently detach actions from their prices"
        )
    for column in ("open", "close"):
        try:
            working[column] = working[column].astype("float64")
        except (TypeError, ValueError) as exc:
            raise DataSourceError(f"{column} contains non-numeric values: {exc}") from exc
        if working[column].isna().any():
            raise DataSourceError(f"{column} contains missing values; nothing is filled or skipped")
        if not (working[column] > 0).all():
            raise DataSourceError(f"{column} must be positive on every row")
    return working.reset_index(drop=True), chronological


def run_simulation(
    sessions: pd.DataFrame,
    actions: Sequence[int],
    settings: SimulationSettings,
) -> dict[str, Any]:
    """Run one action sequence over one price series and return the equity rows and summary."""
    working, _ = _prepare_sessions(sessions)
    if len(actions) != len(working):
        raise ValueError(f"expected {len(working)} actions, got {len(actions)}")

    simulator = PortfolioSimulator(settings)
    rows: list[dict[str, object]] = []
    for session_date, open_price, close_price, action in zip(
        working["session_date"], working["open"], working["close"], actions
    ):
        record = simulator.step(int(action), float(open_price), float(close_price))
        record["session_date"] = str(session_date)
        rows.append(record)

    equity = pd.DataFrame(rows)
    equity = equity[
        [
            "session_date",
            "price_open",
            "price_close",
            "action",
            "action_name",
            "executed",
            "reason",
            "shares_before",
            "shares_after",
            "trade_shares",
            "trade_notional",
            "transaction_cost",
            "cash",
            "position_value",
            "portfolio_value",
        ]
    ]
    return {
        "equity": equity,
        "summary": summary_metrics(equity, settings),
        "final_cash": simulator.cash,
        "final_shares": simulator.shares,
        "total_costs": simulator.total_costs,
    }


def summary_metrics(equity: pd.DataFrame, settings: SimulationSettings) -> dict[str, Any]:
    """Return, trades, costs and drawdown of one completed equity path."""
    values = equity["portfolio_value"].astype("float64").to_numpy()
    running_peak = max(settings.initial_capital, float(values.max())) if len(values) else settings.initial_capital
    # Drawdown is measured against the running peak, starting from the initial capital.
    peaks = []
    peak = settings.initial_capital
    for value in values:
        peak = max(peak, float(value))
        peaks.append(peak)
    drawdowns = [1.0 - value / peak if peak > 0 else 0.0 for value, peak in zip(values, peaks)]
    final_value = float(values[-1]) if len(values) else settings.initial_capital
    return {
        "sessions": int(len(equity)),
        "first_session": str(equity["session_date"].iloc[0]) if len(equity) else None,
        "last_session": str(equity["session_date"].iloc[-1]) if len(equity) else None,
        "initial_capital": round(float(settings.initial_capital), 6),
        "final_value": round(final_value, 6),
        "total_return": round(final_value / settings.initial_capital - 1.0, 6),
        "number_of_trades": int(equity["executed"].sum()),
        "total_transaction_costs": round(float(equity["transaction_cost"].sum()), 6),
        "max_drawdown": round(float(max(drawdowns)) if drawdowns else 0.0, 6),
        "final_cash": round(float(equity["cash"].iloc[-1]) if len(equity) else settings.initial_capital, 6),
        "final_shares": int(equity["shares_after"].iloc[-1]) if len(equity) else 0,
        "peak_value": round(float(running_peak), 6),
    }


def buy_and_hold_actions(session_count: int) -> list[int]:
    """Buy at the first session's open and never trade again."""
    if session_count < 1:
        raise ValueError("buy-and-hold needs at least one session")
    return [ACTION_BUY] + [ACTION_HOLD] * (session_count - 1)


def momentum_actions(sessions: pd.DataFrame, *, window: int = MOMENTUM_WINDOW) -> list[int]:
    """Buy when the previous close is above its trailing mean, sell otherwise.

    Only closes up to ``t-1`` are read, so the action for session ``t`` uses no
    information from ``t`` itself. Redundant actions are harmless: the engine
    records them as no-ops.
    """
    if window < 1:
        raise ValueError(f"momentum window must be >= 1, got {window}")
    closes = sessions["close"].astype("float64").to_numpy()
    actions: list[int] = []
    for index in range(len(closes)):
        if index == 0:
            actions.append(ACTION_HOLD)
            continue
        start = max(0, index - window)
        trailing_mean = float(closes[start:index].mean())
        actions.append(ACTION_BUY if float(closes[index - 1]) > trailing_mean else ACTION_SELL)
    return actions


def _strategy_summary(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"strategy": name, **result["summary"]}


def run_benchmarks(config: Config, *, root: Path | None = None, write: bool = True) -> dict[str, Any]:
    """Simulate buy-and-hold and the momentum rule over the cleaned market table."""
    market_file = market_processed_path(config, root=root)
    if not market_file.exists():
        raise DataNotFetchedError(
            f"no cleaned market table at {market_file}; run: "
            "python -m stock_project.data.inspect_data --config configs/config.example.yaml"
        )

    market = pd.read_csv(market_file)
    timestamps = pd.to_datetime(market["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise DataSourceError("market table contains timestamps that could not be parsed as UTC")
    order = timestamps.sort_values(kind="stable").index
    market = market.loc[order].reset_index(drop=True)
    sessions = pd.DataFrame(
        {
            "session_date": pd.to_datetime(market["timestamp"], utc=True).dt.strftime("%Y-%m-%d"),
            "open": market["open"],
            "close": market["close"],
        }
    )

    settings = settings_from_config(config)
    strategies = {
        "buy_and_hold": {
            "description": "buy at the first session's open with position_fraction of equity, then hold",
            "actions": buy_and_hold_actions(len(sessions)),
        },
        f"momentum_{MOMENTUM_WINDOW}": {
            "description": (
                f"buy when the previous close is above its trailing {MOMENTUM_WINDOW}-session mean, else sell"
            ),
            "actions": momentum_actions(sessions, window=MOMENTUM_WINDOW),
        },
    }

    results: dict[str, dict[str, Any]] = {}
    for name, strategy in strategies.items():
        results[name] = run_simulation(sessions, strategy["actions"], settings)

    equity_parts = []
    for name, result in results.items():
        path = result["equity"].copy()
        path.insert(0, "strategy", name)
        equity_parts.append(path)
    equity_frame = pd.concat(equity_parts, ignore_index=True)

    def _strategy_checks() -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        for name, result in results.items():
            equity = result["equity"]
            rate = settings.cost_rate
            cost_ok = True
            for _, row in equity.loc[equity["executed"]].iterrows():
                expected = float(row["trade_notional"]) * rate
                if abs(float(row["transaction_cost"]) - expected) > 1e-9:
                    cost_ok = False
                    break
            checks.extend(
                [
                    {
                        "name": f"{name}_cash_never_negative",
                        "passed": bool((equity["cash"].astype("float64") >= -1e-9).all()),
                        "detail": f"minimum cash {float(equity['cash'].min())!r}; buys are capped by available cash plus cost",
                    },
                    {
                        "name": f"{name}_shares_never_negative",
                        "passed": bool((equity["shares_after"].astype("int64") >= 0).all()),
                        "detail": "long-only; no short position is ever created",
                    },
                    {
                        "name": f"{name}_costs_match_notional",
                        "passed": cost_ok,
                        "detail": f"every executed trade is charged exactly notional x {settings.cost_rate} ({settings.transaction_cost_bps} bps)",
                    },
                ]
            )
        return checks

    replay = {
        name: run_simulation(sessions, strategy["actions"], settings)["summary"]
        for name, strategy in strategies.items()
    }
    deterministic = all(replay[name] == results[name]["summary"] for name in results)

    checks = [
        {
            "name": "sessions_loaded",
            "passed": len(sessions) > 0,
            "detail": f"{len(sessions)} session(s) read from {relative_to_root(market_file, root)}",
        },
        {
            "name": "sessions_chronological_and_unique",
            "passed": bool(sessions["session_date"].is_monotonic_increasing) and not sessions["session_date"].duplicated().any(),
            "detail": "sessions arrive in ascending order with unique dates; the simulator never reorders rows",
        },
        {
            "name": "prices_complete_and_positive",
            "passed": bool(
                sessions[["open", "close"]].notna().all().all() and (sessions[["open", "close"]] > 0).all().all()
            ),
            "detail": "every session has a positive open and close; nothing is filled",
        },
        {
            "name": "buy_and_hold_enters_once",
            "passed": int(results["buy_and_hold"]["equity"]["executed"].sum()) == 1,
            "detail": "the buy-and-hold benchmark executes exactly one trade",
        },
        {
            "name": "benchmarks_cover_every_session",
            "passed": all(len(result["equity"]) == len(sessions) for result in results.values()),
            "detail": "each strategy produces one equity row per session",
        },
        {
            "name": "replay_is_deterministic",
            "passed": deterministic,
            "detail": "re-running identical prices and actions yields identical cash, shares and portfolio values",
        },
        {
            "name": "maximum_drawdown_in_unit_interval",
            "passed": all(0.0 <= result["summary"]["max_drawdown"] <= 1.0 for result in results.values()),
            "detail": "drawdown is measured against the running peak, including the initial capital",
        },
        *_strategy_checks(),
    ]

    report: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "phase": 5,
        "phase_name": "Trading simulator",
        "config": config.to_dict(),
        "inputs": {
            "market_table": relative_to_root(market_file, root),
            "sessions": int(len(sessions)),
            "first_session": str(sessions["session_date"].iloc[0]),
            "last_session": str(sessions["session_date"].iloc[-1]),
        },
        "execution_rules": {
            "timing": (
                "each session's action is decided pre-open and filled at that session's open; the portfolio is "
                "marked to market at that session's close"
            ),
            "actions": {
                "hold": "do nothing",
                "buy": "open a long position only when flat; invest position_fraction of current equity, whole shares only",
                "sell": "close the entire position",
            },
            "position_limit": "long-only, one position, no pyramiding, no shorting",
            "sizing": (
                f"position_fraction={config.trading.position_fraction} of equity at the fill price, capped by "
                "available cash including the transaction cost; unaffordable sizes are reduced, and zero affordable "
                "shares means no trade"
            ),
            "costs": (
                f"transaction_cost_bps={config.trading.transaction_cost_bps} charged on the traded notional of both "
                "buys and sells"
            ),
            "idealisation": "no slippage, partial fills, liquidity limits, taxes, or dividends; close is the raw (unadjusted) close",
        },
        "strategies": {name: _strategy_summary(name, result) for name, result in results.items()},
        "checks": checks,
        "limitations": [
            "Historical simulation only: this is not a forecast and does not establish future profitability.",
            "One instrument, whole shares, one position, long-only: no portfolio construction or risk limits beyond position_fraction.",
            "Fills occur at the open with no slippage or liquidity model; real execution would be worse.",
            "The transaction cost is a flat bps of notional on both sides; market impact and spread are not modelled separately.",
            "Raw close is used, so dividends and any unadjusted corporate actions are ignored.",
            "Momentum and buy-and-hold are benchmarks for later comparison, not recommended strategies.",
        ],
        "next_step": (
            "Phase 6 — wrap this tested simulator in a Gymnasium environment whose observation uses only information "
            "available at the simulated time, then train and evaluate a small RL agent against these benchmarks."
        ),
    }

    if write:
        ensure_directories(root)
        equity_path = results_dir(root) / EQUITY_FILENAME
        equity_frame.to_csv(equity_path, index=False, lineterminator="\n")
        report_path = results_dir(root) / REPORT_FILENAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", relative_to_root(report_path, root))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate buy-and-hold and momentum benchmarks with the Phase 5 deterministic simulator."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--print-json", action="store_true", help="also print the full report to stdout")
    parser.add_argument("--root", type=Path, default=None, help="project root to read/write (default: this repository)")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config = load_config(args.config)
    root = Path(args.root) if args.root else None
    report = run_benchmarks(config, root=root)

    period = report["inputs"]
    print(
        f"period   : {period['sessions']} sessions {period['first_session']}..{period['last_session']} "
        f"({config.trading.initial_capital:,.0f} initial, {config.trading.transaction_cost_bps} bps)"
    )
    for name, summary in report["strategies"].items():
        print(
            f"{name:<12}: final {summary['final_value']:,.2f}  return {summary['total_return']:+.2%}  "
            f"trades {summary['number_of_trades']:>3}  costs {summary['total_transaction_costs']:,.2f}  "
            f"max drawdown {summary['max_drawdown']:.2%}"
        )
    print(f"report   : {relative_to_root(results_dir(root) / REPORT_FILENAME, root)}")
    failed = [check for check in report["checks"] if not check["passed"]]
    for check in report["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    if args.print_json:
        print(json.dumps(report, indent=2))
    if failed:
        print(f"\n{len(failed)} check(s) failed — the simulation is not reproducible as reported.")
        return 1
    print(f"\nAll {len(report['checks'])} Phase 5 checks passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
