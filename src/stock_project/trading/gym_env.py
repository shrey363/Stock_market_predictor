"""Gymnasium wrapper around the tested Phase 5 simulator (Phase 6).

Information set (invariant 6)
----------------------------
The observation for session ``t`` is ``[feature row t, holding flag, cash fraction]``.
The feature row is built by Phases 3–4 from bars up to ``t-1`` only and from news
whose ``available_from`` is before session ``t``'s cutoff, so nothing from session
``t`` itself or later is visible. The two state values describe the current
portfolio and are known at the decision point too.

Environment contract
--------------------
* ``action_space``: ``Discrete(3)`` — hold (0), buy (1), sell (2), reusing the
  simulator's action ids.
* Execution and accounting are delegated to :class:`PortfolioSimulator`, so the
  cash/shares/cost arithmetic is the unit-tested Phase 5 code, not a copy.
* Reward is the step's simple portfolio return, ``value_t / value_{t-1} - 1``,
  with ``value_0`` = initial capital. Costs are therefore already inside the
  reward (they reduce cash at the fill).
* An episode is one pass over the supplied sessions and ends with
  ``truncated=True`` (time limit); the environment never terminates early.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..common.errors import DataSourceError
from .simulator import (
    ACTION_NAMES,
    PortfolioSimulator,
    SimulationSettings,
    _prepare_sessions,
)

#: State values appended to each feature row: holding flag, cash fraction.
STATE_DIM = 2

OBSERVATION_DESCRIPTION = (
    "the session's pre-cutoff feature row (market features from bars up to t-1 plus news features "
    "available before the session's cutoff), followed by [holding flag, cash fraction]"
)


class TradingEnv(gym.Env[np.ndarray, int]):
    """One episode = one chronological pass over the supplied sessions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        observations: np.ndarray | pd.DataFrame,
        sessions: pd.DataFrame,
        settings: SimulationSettings,
        *,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        matrix = np.asarray(observations, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError(f"observations must be a non-empty 2-D array, got shape {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise DataSourceError("observations contain missing or non-finite values; nothing is imputed")

        working, _ = _prepare_sessions(sessions)  # validates prices and chronology
        if len(working) != len(matrix):
            raise ValueError(f"got {len(matrix)} observation row(s) but {len(working)} session(s)")

        self.feature_dim = int(matrix.shape[1])
        self.observations = matrix
        self.session_dates = working["session_date"].to_numpy(dtype=str)
        self.opens = working["open"].to_numpy(dtype=float)
        self.closes = working["close"].to_numpy(dtype=float)
        self.settings = settings
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_dim + STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        self.simulator = PortfolioSimulator(settings)
        self._index = 0
        self._value = float(settings.initial_capital)
        self._cash_fraction = 1.0

    @property
    def horizon(self) -> int:
        return len(self.session_dates)

    @property
    def portfolio_value(self) -> float:
        """Value after the most recent step (initial capital before the first one)."""
        return self._value

    def _observation(self, index: int) -> np.ndarray:
        state = np.array(
            [1.0 if self.simulator.shares > 0 else 0.0, self._cash_fraction], dtype=np.float32
        )
        return np.concatenate([self.observations[index], state]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.simulator = PortfolioSimulator(self.settings)
        self._index = 0
        self._value = float(self.settings.initial_capital)
        self._cash_fraction = 1.0
        return self._observation(0), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._index >= self.horizon:
            raise RuntimeError("episode is over; call reset() before stepping again")
        action = int(action)
        record = self.simulator.step(action, self.opens[self._index], self.closes[self._index])

        value = float(record["portfolio_value"])
        reward = value / self._value - 1.0
        self._value = value
        self._cash_fraction = float(record["cash"]) / value if value > 0 else 0.0
        session_date = self.session_dates[self._index]
        self._index += 1

        truncated = self._index >= self.horizon
        observation = self._observation(self.horizon - 1 if truncated else self._index)
        info = {
            "session_date": session_date,
            "action": action,
            "action_name": record["action_name"],
            "executed": bool(record["executed"]),
            "reason": record["reason"],
            "transaction_cost": float(record["transaction_cost"]),
            "portfolio_value": value,
        }
        return observation, float(reward), False, truncated, info


__all__ = ["OBSERVATION_DESCRIPTION", "STATE_DIM", "TradingEnv"]
