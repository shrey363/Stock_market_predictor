"""Phase 6 Gymnasium environment: contract, accounting reuse and leakage guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from stock_project.common.errors import DataSourceError
from stock_project.trading.gym_env import TradingEnv
from stock_project.trading.simulator import ACTION_BUY, ACTION_HOLD, SimulationSettings


def settings(**overrides) -> SimulationSettings:
    values = {"initial_capital": 1000.0, "position_fraction": 1.0, "transaction_cost_bps": 0.0}
    values.update(overrides)
    return SimulationSettings(**values)


def sessions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "open": [10.0, 12.0, 11.0],
            "close": [11.0, 10.0, 13.0],
        }
    )


def observations() -> np.ndarray:
    return np.array([[0.1, -0.2], [0.2, -0.1], [0.3, 0.0]], dtype=np.float32)


def test_gymnasium_contract_and_episode_boundary():
    env = TradingEnv(observations(), sessions(), settings())
    check_env(env)

    observation, info = env.reset(seed=7)
    assert observation.tolist() == pytest.approx([0.1, -0.2, 0.0, 1.0])
    assert info == {}

    _, reward, terminated, truncated, info = env.step(ACTION_HOLD)
    assert reward == pytest.approx(0.0)
    assert not terminated and not truncated
    assert info["session_date"] == "2020-01-02"

    env.step(ACTION_HOLD)
    _, _, terminated, truncated, _ = env.step(ACTION_HOLD)
    assert not terminated and truncated
    with pytest.raises(RuntimeError, match="episode is over"):
        env.step(ACTION_HOLD)


def test_reward_and_accounting_are_the_phase_five_simulator_values():
    env = TradingEnv(observations(), sessions(), settings(transaction_cost_bps=100.0))
    env.reset()

    observation, reward, _, _, info = env.step(ACTION_BUY)

    # Phase 5 arithmetic: 99 shares at 10, 9.90 cost, 0.10 cash; marked at 11.
    assert info["portfolio_value"] == pytest.approx(1089.1)
    assert info["transaction_cost"] == pytest.approx(9.9)
    assert reward == pytest.approx(1089.1 / 1000.0 - 1.0)
    assert observation[-2] == pytest.approx(1.0)  # holding flag
    assert observation[-1] == pytest.approx(0.1 / 1089.1)  # cash fraction


def test_future_rows_cannot_change_the_current_observation_or_transition():
    baseline = TradingEnv(observations(), sessions(), settings())
    base_observation, _ = baseline.reset()
    _, base_reward, _, _, base_info = baseline.step(ACTION_BUY)

    future_observations = observations()
    future_observations[1:] = 999.0
    future_sessions = sessions()
    future_sessions.loc[1:, ["open", "close"]] = 999.0
    mutated = TradingEnv(future_observations, future_sessions, settings())
    mutated_observation, _ = mutated.reset()
    _, mutated_reward, _, _, mutated_info = mutated.step(ACTION_BUY)

    np.testing.assert_array_equal(mutated_observation, base_observation)
    assert mutated_reward == pytest.approx(base_reward)
    assert mutated_info["portfolio_value"] == pytest.approx(base_info["portfolio_value"])


def test_current_session_prices_are_not_in_the_observation():
    base = TradingEnv(observations(), sessions(), settings())
    base_observation, _ = base.reset()

    changed_sessions = sessions()
    changed_sessions.loc[0, ["open", "close"]] = [100.0, 110.0]
    changed = TradingEnv(observations(), changed_sessions, settings())
    changed_observation, _ = changed.reset()

    np.testing.assert_array_equal(changed_observation, base_observation)


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError, match="observation row"):
        TradingEnv(observations()[:2], sessions(), settings())
    with pytest.raises(DataSourceError, match="non-finite"):
        bad = observations()
        bad[0, 0] = np.nan
        TradingEnv(bad, sessions(), settings())
