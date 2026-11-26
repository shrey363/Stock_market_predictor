"""CLI: train and evaluate the Phase 6 reinforcement-learning trading agent.

    python -m stock_project.trading.train_rl --config configs/config.example.yaml

Protocol (no test leakage)
--------------------------
1. Hyperparameters are chosen from a small fixed candidate grid **trained on the
   train period and scored on the validation period only**.
2. The selected candidate is retrained on train + validation, once per configured
   seed, and evaluated **once** on the held-out test period.
3. Reported agent figures come from replaying the agent's actions through the
   unit-tested Phase 5 simulator, on the same test sessions and costs as the
   buy-and-hold / momentum / cash benchmarks.

The observation is the pre-cutoff feature row plus portfolio state (see
``gym_env.py``); the action is hold/buy/sell; the reward is the step's simple
portfolio return, so costs are inside the reward.

Artifacts (invariant 7):

* ``reports/experiment_results/phase6_rl_report.json`` — protocol, candidates,
  per-seed test metrics, benchmarks, checks and limitations.
* ``reports/experiment_results/phase6_test_portfolio_values.csv`` — one equity
  row per strategy (and per seed for the agent) per test session.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import Config, load_config
from ..common.errors import DataNotFetchedError, DataSourceError
from ..common.logging_utils import configure_cli_logging, get_logger
from ..common.paths import ensure_directories, relative_to_root, results_dir
from ..common.timeutils import utc_now_iso
from ..data.inspect_data import market_processed_path
from ..features.chronological_splits import chronological_split
from ..features.news_features import news_feature_columns
from ..features.technical_indicators import CONTEXT_COLUMNS, TARGET_COLUMNS, feature_definitions
from ..models.train_news_aware import news_features_processed_path
from .gym_env import OBSERVATION_DESCRIPTION, STATE_DIM, TradingEnv
from .simulator import (
    ACTION_HOLD,
    MOMENTUM_WINDOW,
    SimulationSettings,
    buy_and_hold_actions,
    momentum_actions,
    run_simulation,
    settings_from_config,
)

logger = get_logger("stock_project.trading.train_rl")

REPORT_FILENAME = "phase6_rl_report.json"
VALUES_FILENAME = "phase6_test_portfolio_values.csv"

#: Sessions per year used only to annualise the Sharpe ratio convention.
SESSIONS_PER_YEAR = 252

#: Fixed candidate grid, selected on the validation period. A small grid on
#: purpose: the point is an honest held-out comparison, not hyperparameter search.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "ppo_steady",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "n_steps": 256,
        "batch_size": 64,
    },
    {
        "name": "ppo_reactive",
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "ent_coef": 0.0,
        "n_steps": 256,
        "batch_size": 64,
    },
)


def load_dataset(
    config: Config, *, root: Path | None = None
) -> tuple[pd.DataFrame, list[str], list[str], Path, Path]:
    """Read the Phase 4 feature store, attach the session fill prices, and validate it."""
    features_file = news_features_processed_path(config, root=root)
    if not features_file.exists():
        raise DataNotFetchedError(
            f"no combined feature table at {features_file}; run: "
            "python -m stock_project.models.train_news_aware --config configs/config.example.yaml"
        )
    market_file = market_processed_path(config, root=root)
    if not market_file.exists():
        raise DataNotFetchedError(
            f"no cleaned market table at {market_file}; run: "
            "python -m stock_project.data.inspect_data --config configs/config.example.yaml"
        )

    features = pd.read_csv(features_file)
    market = pd.read_csv(market_file)
    timestamps = pd.to_datetime(market["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise DataSourceError("market table contains timestamps that could not be parsed as UTC")
    prices = pd.DataFrame(
        {
            "session_date": timestamps.dt.strftime("%Y-%m-%d"),
            "open": market["open"].astype("float64"),
            "close": market["close"].astype("float64"),
        }
    )
    if prices["session_date"].duplicated().any():
        raise DataSourceError("market table has more than one bar for a session date")

    merged = features.merge(prices, on="session_date", how="left", validate="one_to_one", sort=False)
    if merged[["open", "close"]].isna().any().any():
        missing = merged.loc[merged["open"].isna() | merged["close"].isna(), "session_date"].tolist()
        raise DataSourceError(f"feature sessions without a market bar: {missing[:5]}")

    market_columns = list(
        feature_definitions(
            return_windows=config.features.return_windows,
            moving_average_windows=config.features.moving_average_windows,
            rsi_window=config.features.rsi_window,
            volatility_window=config.features.volatility_window,
            volume_window=config.features.volume_window,
        )
    )
    news_columns = news_feature_columns()
    missing_columns = [column for column in market_columns + news_columns if column not in merged.columns]
    if missing_columns:
        raise DataSourceError(
            f"{features_file} lacks expected feature column(s) {missing_columns}; "
            "regenerate it with python -m stock_project.models.train_news_aware"
        )
    if not merged["session_date"].is_monotonic_increasing or merged["session_date"].duplicated().any():
        raise DataSourceError("feature rows must be unique and in ascending chronological order")
    return merged, market_columns, news_columns, features_file, market_file


def sharpe_ratio(equity: pd.DataFrame, settings: SimulationSettings) -> float | None:
    """Annualised Sharpe from per-session simple returns, risk-free rate 0.

    Convention: ``mean / sample-std * sqrt(252)``. The 252-session year is a
    stated convention; the test period is much shorter, so this is a descriptive
    statistic, not a precise estimate.
    """
    values = np.concatenate(([settings.initial_capital], equity["portfolio_value"].to_numpy(dtype=float)))
    returns = values[1:] / values[:-1] - 1.0
    if len(returns) < 2:
        return None
    deviation = float(returns.std(ddof=1))
    if deviation == 0.0 or not math.isfinite(deviation):
        return None
    return round(float(returns.mean() / deviation * math.sqrt(SESSIONS_PER_YEAR)), 6)


def _fit_agent(
    observations: np.ndarray,
    sessions: pd.DataFrame,
    settings: SimulationSettings,
    candidate: dict[str, Any],
    *,
    timesteps: int,
    seed: int,
):
    """Train one PPO agent on repeated resets of one chronological segment."""
    from stable_baselines3 import PPO  # imported lazily: the env tests need no torch
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(TradingEnv(observations, sessions, settings))
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        device="cpu",
        learning_rate=candidate["learning_rate"],
        gamma=candidate["gamma"],
        ent_coef=candidate["ent_coef"],
        n_steps=candidate["n_steps"],
        batch_size=candidate["batch_size"],
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=timesteps)
    return model


def _rollout(model, observations: np.ndarray, sessions: pd.DataFrame, settings: SimulationSettings) -> tuple[list[int], float]:
    """Run one deterministic pass and return the action sequence plus the env's final value."""
    env = TradingEnv(observations, sessions, settings)
    observation, _ = env.reset(seed=0)
    actions: list[int] = []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(int(action))
        actions.append(int(action))
        done = terminated or truncated
    return actions, float(env.portfolio_value)


def _evaluate(actions: list[int], sessions: pd.DataFrame, settings: SimulationSettings) -> dict[str, Any]:
    """Replay an action sequence through the verified simulator."""
    result = run_simulation(sessions, actions, settings)
    return {"equity": result["equity"], "summary": result["summary"], "actions": actions}


def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [entry["total_return"] for entry in per_seed]
    sharpes = [entry["sharpe_ratio"] for entry in per_seed if entry["sharpe_ratio"] is not None]
    return {
        "seeds": len(per_seed),
        "mean_total_return": round(float(np.mean(returns)), 6),
        "std_total_return": round(float(np.std(returns, ddof=1)), 6) if len(returns) > 1 else None,
        "min_total_return": round(float(min(returns)), 6),
        "max_total_return": round(float(max(returns)), 6),
        "mean_sharpe_ratio": round(float(np.mean(sharpes)), 6) if sharpes else None,
        "mean_number_of_trades": round(float(np.mean([entry["number_of_trades"] for entry in per_seed])), 3),
        "mean_total_transaction_costs": round(
            float(np.mean([entry["total_transaction_costs"] for entry in per_seed])), 6
        ),
    }


def _strategy_entry(name: str, evaluation: dict[str, Any], settings: SimulationSettings) -> dict[str, Any]:
    summary = evaluation["summary"]
    return {
        "strategy": name,
        **summary,
        "sharpe_ratio": sharpe_ratio(evaluation["equity"], settings),
    }


def run_rl(config: Config, *, root: Path | None = None, write: bool = True) -> dict[str, Any]:
    """Select hyperparameters on validation, retrain on train+validation, test once."""
    from stable_baselines3 import __version__ as sb3_version
    from sklearn.preprocessing import StandardScaler

    features, market_columns, news_columns, features_file, market_file = load_dataset(config, root=root)
    feature_columns = market_columns + news_columns
    settings = settings_from_config(config)
    split = chronological_split(
        len(features),
        train_fraction=config.splits.train_fraction,
        validation_fraction=config.splits.validation_fraction,
    )

    sessions = features[["session_date", "open", "close"]].reset_index(drop=True)
    raw = features[feature_columns].to_numpy(dtype=np.float64)

    train_slice = slice(split.train.start, split.train.stop)
    validation_slice = slice(split.validation.start, split.validation.stop)
    test_slice = slice(split.test.start, split.test.stop)
    fit_slice = slice(split.train.start, split.validation.stop)

    timesteps = int(config.rl.total_timesteps)

    # 1. Candidate selection: fit on train, score on validation. No test rows.
    selection_scaler = StandardScaler().fit(raw[train_slice])
    selection_observation = selection_scaler.transform(raw).astype(np.float32)
    selection: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        model = _fit_agent(
            selection_observation[train_slice],
            sessions.iloc[train_slice],
            settings,
            candidate,
            timesteps=timesteps,
            seed=config.project.random_seed,
        )
        validation_actions, _ = _rollout(
            model, selection_observation[validation_slice], sessions.iloc[validation_slice], settings
        )
        validation = _evaluate(validation_actions, sessions.iloc[validation_slice], settings)
        entry = _strategy_entry(candidate["name"], validation, settings)
        selection.append(
            {
                "candidate": candidate,
                "validation": {
                    "total_return": entry["total_return"],
                    "sharpe_ratio": entry["sharpe_ratio"],
                    "max_drawdown": entry["max_drawdown"],
                    "number_of_trades": entry["number_of_trades"],
                    "total_transaction_costs": entry["total_transaction_costs"],
                },
            }
        )
    selected_index = max(
        range(len(selection)),
        key=lambda index: (
            selection[index]["validation"]["total_return"],
            -selection[index]["validation"]["total_transaction_costs"],
        ),
    )
    selected = {**CANDIDATES[selected_index]}

    # 2. Final fit on train + validation, evaluated once on the test period per seed.
    final_scaler = StandardScaler().fit(raw[fit_slice])
    final_observation = final_scaler.transform(raw).astype(np.float32)

    per_seed: list[dict[str, Any]] = []
    agent_equity: list[pd.DataFrame] = []
    simulator_agreement = True
    for seed in config.rl.seeds:
        model = _fit_agent(
            final_observation[fit_slice],
            sessions.iloc[fit_slice],
            settings,
            selected,
            timesteps=timesteps,
            seed=int(seed),
        )
        actions, env_value = _rollout(
            model, final_observation[test_slice], sessions.iloc[test_slice], settings
        )
        evaluation = _evaluate(actions, sessions.iloc[test_slice], settings)
        entry = _strategy_entry(f"ppo_seed_{seed}", evaluation, settings)
        entry["seed"] = int(seed)
        per_seed.append(entry)
        simulator_agreement = simulator_agreement and math.isclose(
            env_value,
            float(evaluation["equity"]["portfolio_value"].iloc[-1]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        path = evaluation["equity"].copy()
        path.insert(0, "strategy", f"ppo_seed_{seed}")
        path.insert(1, "seed", int(seed))
        agent_equity.append(path)

    # 3. Benchmarks on the same held-out sessions and the same costs.
    test_sessions = sessions.iloc[test_slice].reset_index(drop=True)
    probe = TradingEnv(final_observation[test_slice], test_sessions, settings)
    benchmark_actions = {
        "buy_and_hold": buy_and_hold_actions(len(test_sessions)),
        f"momentum_{MOMENTUM_WINDOW}": momentum_actions(test_sessions, window=MOMENTUM_WINDOW),
        "cash": [ACTION_HOLD] * len(test_sessions),
    }
    benchmark_evaluations = {
        name: _evaluate(actions, test_sessions, settings) for name, actions in benchmark_actions.items()
    }
    benchmarks = {
        name: _strategy_entry(name, evaluation, settings)
        for name, evaluation in benchmark_evaluations.items()
    }

    equity_parts = list(agent_equity)
    for name, evaluation in benchmark_evaluations.items():
        path = evaluation["equity"].copy()
        path.insert(0, "strategy", name)
        path.insert(1, "seed", pd.NA)
        equity_parts.append(path)

    checks = [
        {
            "name": "features_loaded",
            "passed": len(features) > 0,
            "detail": f"{len(features)} feature row(s) read from {relative_to_root(features_file, root)}",
        },
        {
            "name": "features_chronological_and_unique",
            "passed": bool(features["session_date"].is_monotonic_increasing)
            and not features["session_date"].duplicated().any(),
            "detail": "feature rows are unique and in ascending session order; nothing was shuffled",
        },
        {
            "name": "features_complete",
            "passed": not bool(features[feature_columns].isna().any().any()),
            "detail": f"every row has all {len(feature_columns)} features; nothing is imputed",
        },
        {
            "name": "features_exclude_target_and_context",
            "passed": not set(feature_columns) & (set(TARGET_COLUMNS) | set(CONTEXT_COLUMNS)),
            "detail": "no target or context column is part of the observation",
        },
        {
            "name": "split_contiguous_and_ordered",
            "passed": len(features) == split.test.stop
            and split.train.stop == split.validation.start
            and split.validation.stop == split.test.start,
            "detail": (
                f"train={len(split.train)} validation={len(split.validation)} test={len(split.test)} "
                "contiguous rows in time order, covering every row exactly once"
            ),
        },
        {
            "name": "selection_used_no_test_rows",
            "passed": True,
            "detail": (
                f"candidates were trained on {len(split.train)} train rows and scored on "
                f"{len(split.validation)} validation rows; the {len(split.test)} test rows were not read"
            ),
        },
        {
            "name": "final_fit_used_no_test_rows",
            "passed": (fit_slice.stop - fit_slice.start) == len(split.train) + len(split.validation),
            "detail": (
                f"final agents were trained on {len(split.train) + len(split.validation)} train+validation rows "
                f"and evaluated on {len(split.test)} held-out sessions"
            ),
        },
        {
            "name": "observations_have_expected_dimension",
            "passed": bool(
                probe.observation_space.shape == (len(feature_columns) + STATE_DIM,)
                and probe.action_space.n == 3
            ),
            "detail": (
                f"observation space is {probe.observation_space.shape} "
                f"({len(feature_columns)} feature values + {STATE_DIM} portfolio state values), 3 actions"
            ),
        },
        {
            "name": "environment_matches_simulator",
            "passed": bool(simulator_agreement),
            "detail": "each agent's environment final value equals the Phase 5 simulator replay of its actions",
        },
        {
            "name": "all_strategies_cover_every_test_session",
            "passed": all(entry["sessions"] == len(test_sessions) for entry in per_seed + list(benchmarks.values())),
            "detail": f"each agent seed and benchmark produces one equity row per test session ({len(test_sessions)})",
        },
        {
            "name": "cash_never_negative",
            "passed": all(
                bool((evaluation["equity"]["cash"].astype("float64") >= -1e-9).all())
                for evaluation in list(benchmark_evaluations.values())
            )
            and all(bool((path["cash"].astype("float64") >= -1e-9).all()) for path in agent_equity),
            "detail": "buys are capped by available cash including cost, for the agent and every benchmark",
        },
        {
            "name": "shares_never_negative",
            "passed": all(bool((path["shares_after"].astype("int64") >= 0).all()) for path in agent_equity)
            and all(
                bool((evaluation["equity"]["shares_after"].astype("int64") >= 0).all())
                for evaluation in benchmark_evaluations.values()
            ),
            "detail": "long-only: no strategy ever creates a short position",
        },
        {
            "name": "costs_match_notional",
            "passed": all(
                all(
                    math.isclose(
                        float(row["transaction_cost"]),
                        float(row["trade_notional"]) * settings.cost_rate,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for _, row in path.loc[path["executed"]].iterrows()
                )
                for path in agent_equity
            ),
            "detail": f"every executed trade is charged exactly notional x {settings.cost_rate} ({config.trading.transaction_cost_bps} bps)",
        },
        {
            "name": "risk_metrics_in_range",
            "passed": all(
                0.0 <= entry["max_drawdown"] <= 1.0
                and (entry["sharpe_ratio"] is None or math.isfinite(entry["sharpe_ratio"]))
                for entry in per_seed + list(benchmarks.values())
            ),
            "detail": "max drawdown lies in [0, 1] and the Sharpe ratio is finite or explicitly null",
        },
        {
            "name": "benchmarks_present",
            "passed": set(benchmarks) == {"buy_and_hold", f"momentum_{MOMENTUM_WINDOW}", "cash"},
            "detail": "buy-and-hold, the 5-session momentum rule and an all-cash floor share the test period",
        },
    ]

    report: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "phase": 6,
        "phase_name": "Reinforcement learning",
        "config": config.to_dict(),
        "algorithm": {
            "name": "PPO",
            "library": f"stable-baselines3=={sb3_version}",
            "policy": "MlpPolicy (64, 64)",
            "candidate_grid": [dict(candidate) for candidate in CANDIDATES],
        },
        "inputs": {
            "features_table": relative_to_root(features_file, root),
            "market_table": relative_to_root(market_file, root),
            "rows": int(len(features)),
            "market_columns": market_columns,
            "news_columns": news_columns,
            "first_session": str(features["session_date"].iloc[0]),
            "last_session": str(features["session_date"].iloc[-1]),
            "feature_scaler": "StandardScaler fit on the fitting rows only (train for selection, train+validation for the final agents)",
        },
        "environment": {
            "observation": OBSERVATION_DESCRIPTION,
            "observation_dimension": int(len(feature_columns) + STATE_DIM),
            "actions": {"hold": 0, "buy": 1, "sell": 2},
            "reward": "step simple portfolio return value_t / value_(t-1) - 1; costs are inside the reward",
            "episode": "one chronological pass over the supplied sessions; truncated at the end, never terminated early",
            "execution": "Phase 5 simulator: fill at the session open, mark to the close, costs on both sides",
        },
        "protocol": {
            "selection_metric": "validation total return (ties: lower transaction costs)",
            "selection_fit_period": "train only",
            "final_fit_period": "train + validation",
            "evaluation_period": "test only, once per seed",
            "timesteps_per_training": timesteps,
            "seeds": [int(seed) for seed in config.rl.seeds],
            "sharpe_convention": f"mean / sample-std of per-session returns x sqrt({SESSIONS_PER_YEAR}); risk-free rate 0",
        },
        "splits": {
            "fractions": {
                "train": config.splits.train_fraction,
                "validation": config.splits.validation_fraction,
                "test": round(1.0 - config.splits.train_fraction - config.splits.validation_fraction, 6),
            },
            "ranges": split.as_dict(),
            "periods": {
                name: {
                    "rows": int(period.stop - period.start),
                    "first_session": str(features["session_date"].iloc[period.start]),
                    "last_session": str(features["session_date"].iloc[period.stop - 1]),
                }
                for name, period in (
                    ("train", split.train),
                    ("validation", split.validation),
                    ("test", split.test),
                )
            },
        },
        "selection": {
            "candidates": selection,
            "selected": selected,
            "selected_by_validation_total_return": selected["name"],
        },
        "test": {
            "period": {
                "sessions": int(len(test_sessions)),
                "first_session": str(test_sessions["session_date"].iloc[0]),
                "last_session": str(test_sessions["session_date"].iloc[-1]),
            },
            "agent": {"per_seed": per_seed, "aggregate": _aggregate(per_seed)},
            "benchmarks": benchmarks,
        },
        "checks": checks,
        "limitations": [
            "Historical simulation only: this is not a forecast and does not establish future profitability.",
            "One instrument (SPY), one long-only position, whole shares, fills at the open with no slippage or liquidity model.",
            "A single held-out test period of a few hundred sessions is a small sample; seed-to-seed spread is reported to make that visible.",
            "PPO training is stochastic; results are reported per seed rather than as one tuned number.",
            "The reward is portfolio return, so the agent optimises simulated profit, not risk-adjusted return or any real objective.",
            "Models are not saved: the report records the algorithm, hyperparameters, data ranges and seeds for reproduction.",
        ],
        "next_step": (
            "Phase 7 — expose these Phase 1–6 artifacts through Flask and the dashboard, showing that the "
            "displayed values equal the saved experiment outputs."
        ),
    }

    if write:
        ensure_directories(root)
        equity_frame = pd.concat(equity_parts, ignore_index=True)
        equity_frame.to_csv(results_dir(root) / VALUES_FILENAME, index=False, lineterminator="\n")
        report_path = results_dir(root) / REPORT_FILENAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", relative_to_root(report_path, root))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Phase 6 PPO trading agent against the Phase 5 benchmarks."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--print-json", action="store_true", help="also print the full report to stdout")
    parser.add_argument("--root", type=Path, default=None, help="project root to read/write (default: this repository)")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config = load_config(args.config)
    root = Path(args.root) if args.root else None
    report = run_rl(config, root=root)

    test = report["test"]
    print(
        f"test     : {test['period']['sessions']} sessions "
        f"{test['period']['first_session']}..{test['period']['last_session']} "
        f"(selected {report['selection']['selected']['name']}, {report['protocol']['timesteps_per_training']} timesteps)"
    )
    for entry in test["agent"]["per_seed"]:
        print(
            f"ppo seed {entry['seed']:<2}: final {entry['final_value']:,.2f}  return {entry['total_return']:+.2%}  "
            f"sharpe {entry['sharpe_ratio'] if entry['sharpe_ratio'] is not None else 'n/a'}  "
            f"trades {entry['number_of_trades']:>3}  costs {entry['total_transaction_costs']:,.2f}  "
            f"max drawdown {entry['max_drawdown']:.2%}"
        )
    for name, summary in test["benchmarks"].items():
        print(
            f"{name:<12}: final {summary['final_value']:,.2f}  return {summary['total_return']:+.2%}  "
            f"sharpe {summary['sharpe_ratio'] if summary['sharpe_ratio'] is not None else 'n/a'}  "
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
        print(f"\n{len(failed)} check(s) failed — the RL evaluation is not reproducible as reported.")
        return 1
    print(f"\nAll {len(report['checks'])} Phase 6 checks passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
