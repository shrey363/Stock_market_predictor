"""Market features: hand-computed indicator values and the prediction cutoff.

Every expected number is derived by hand from a tiny price series; the leakage
tests are real assertions (mutating future prices must not change a row, while
mutating the predicted session must change only its target).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_project.common.errors import DataSourceError
from stock_project.features.technical_indicators import (
    CONTEXT_COLUMNS,
    TARGET_COLUMNS,
    build_market_features,
    feature_definitions,
    intraday_range,
    overnight_gap,
    rolling_volatility,
    rsi,
    simple_returns,
    sma_ratio,
    volume_ratio,
)


def _market(values: list[float], *, start: str = "2020-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(values), freq="B", tz="UTC") + pd.Timedelta(hours=5)
    close = pd.Series([float(value) for value in values])
    return pd.DataFrame(
        {
            "symbol": "SPY",
            "timestamp": timestamps,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "adj_close": close,
            "volume": 1_000,
            "source": "unit-test fixture (synthetic, not market data)",
        }
    )


def test_simple_returns_are_exact():
    values = simple_returns(pd.Series([100.0, 110.0, 99.0]), 1)
    assert pd.isna(values.iloc[0])
    assert values.iloc[1] == pytest.approx(0.10)
    assert values.iloc[2] == pytest.approx(-0.10)


def test_sma_ratio_is_distance_from_the_moving_average():
    values = sma_ratio(pd.Series([1.0, 2.0, 3.0]), 2)
    assert pd.isna(values.iloc[0])
    assert values.iloc[1] == pytest.approx(2.0 / 1.5 - 1.0)
    assert values.iloc[2] == pytest.approx(3.0 / 2.5 - 1.0)


def test_rsi_matches_hand_computation():
    """Closes 10, 12, 11, 13 with window 2: gains 2/2 then 1.5, losses 0.5 then 0.25."""
    values = rsi(pd.Series([10.0, 12.0, 11.0, 13.0]), 2)
    assert pd.isna(values.iloc[0]) and pd.isna(values.iloc[1])
    assert values.iloc[2] == pytest.approx(100.0 - 100.0 / 3.0)  # RS = 1.0 / 0.5
    assert values.iloc[3] == pytest.approx(100.0 - 100.0 / 7.0)  # RS = 1.5 / 0.25


def test_rsi_edge_cases_are_reported_not_guessed():
    assert rsi(pd.Series([1.0, 2.0, 3.0, 4.0]), 2).iloc[-1] == pytest.approx(100.0)
    assert rsi(pd.Series([4.0, 3.0, 2.0, 1.0]), 2).iloc[-1] == pytest.approx(0.0)
    # A flat price has no up or down movement: the ratio is undefined, reported as 50.
    assert rsi(pd.Series([5.0, 5.0, 5.0, 5.0]), 2).iloc[-1] == pytest.approx(50.0)


def test_rolling_volatility_is_sample_stdev_of_returns():
    close = pd.Series([100.0, 110.0, 99.0, 108.9])
    values = rolling_volatility(close, 3)
    assert values.iloc[3] == pytest.approx(float(np.std([0.1, -0.1, 0.1], ddof=1)))


def test_volume_ratio_is_relative_to_the_trailing_mean():
    values = volume_ratio(pd.Series([100.0, 100.0, 300.0]), 2)
    assert values.iloc[2] == pytest.approx(1.5)


def test_intraday_range_and_overnight_gap_are_exact():
    close = pd.Series([100.0, 110.0])
    high = pd.Series([102.0, 111.0])
    low = pd.Series([99.0, 108.0])
    open_ = pd.Series([100.0, 105.0])
    assert intraday_range(high, low, close).iloc[1] == pytest.approx((111.0 - 108.0) / 110.0)
    assert overnight_gap(open_, close).iloc[1] == pytest.approx(105.0 / 100.0 - 1.0)


def test_invalid_windows_are_rejected():
    with pytest.raises(ValueError):
        rsi(pd.Series([1.0, 2.0, 3.0]), 1)
    with pytest.raises(ValueError):
        sma_ratio(pd.Series([1.0, 2.0, 3.0]), 1)
    with pytest.raises(ValueError):
        simple_returns(pd.Series([1.0, 2.0]), 0)


def test_feature_table_contract_and_warmup_drop(synthetic_market):
    frame = synthetic_market(120)
    features, quality = build_market_features(frame, moving_average_windows=(10, 50))
    definitions = feature_definitions(moving_average_windows=(10, 50))

    assert list(features.columns) == list(CONTEXT_COLUMNS) + list(definitions) + list(TARGET_COLUMNS)
    # sma_ratio_50 needs 50 bars and is shifted forward one row, so rows 0..49 are undefined.
    assert quality.rows_dropped_missing_required == 50
    assert len(features) == 120 - 50
    assert not features[list(definitions)].isna().any().any()
    assert features["session_date"].is_monotonic_increasing
    assert set(features["target"].unique()) <= {0, 1}


def test_target_and_cutoff_columns_follow_the_documented_rule(synthetic_market):
    frame = synthetic_market(80)
    features, _ = build_market_features(
        frame, return_windows=(1,), moving_average_windows=(2,), rsi_window=2, volatility_window=2, volume_window=2
    )
    warmup = len(frame) - len(features)
    closes = frame.sort_values("timestamp")["close"].to_numpy()

    # Row i predicts market session i + warmup, so its target is that session's own move
    # relative to the previous close (the "next session" from the pre-open decision point).
    for row in range(len(features)):
        session = row + warmup
        expected = 1 if closes[session] > closes[session - 1] else 0
        assert features["target"].iloc[row] == expected

    first = features.iloc[0]
    assert first["decision_time_utc"].endswith("T05:00:00Z")
    assert first["news_cutoff_utc"] == f"{first['session_date']}T00:00:00Z"

    later_cutoff, _ = build_market_features(
        frame,
        return_windows=(1,),
        moving_average_windows=(2,),
        rsi_window=2,
        volatility_window=2,
        volume_window=2,
        cutoff_time="13:30",
    )
    assert later_cutoff.iloc[0]["news_cutoff_utc"] == f"{later_cutoff.iloc[0]['session_date']}T13:30:00Z"


def test_mutating_the_predicted_session_leaves_its_features_untouched(synthetic_market):
    """Invariant 2: no value from session t or later may appear in row t."""
    frame = synthetic_market(140, seed=3)
    features, _ = build_market_features(frame, moving_average_windows=(10, 50))
    feature_columns = [column for column in features.columns if column not in (*CONTEXT_COLUMNS, *TARGET_COLUMNS)]

    session = str(features["session_date"].iloc[80])
    before = features.set_index("session_date").loc[session]

    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    position = int((ordered["timestamp"].dt.strftime("%Y-%m-%d") == session).to_numpy().nonzero()[0][0])
    mutated = ordered.copy()
    mutated.loc[position:, ["open", "high", "low", "close"]] = mutated.loc[position:, ["open", "high", "low", "close"]] * 1.5

    after_features, _ = build_market_features(mutated, moving_average_windows=(10, 50))
    after = after_features.set_index("session_date").loc[session]

    for column in feature_columns:
        assert after[column] == pytest.approx(before[column]), f"{column} changed with future prices"

    # The target *does* use the predicted session's own price, so the guard above is
    # not vacuous: multiplying that close by 1.5 must change the realised return.
    assert after["target"] == 1
    assert after["next_session_return"] != pytest.approx(before["next_session_return"])


def test_features_do_depend_on_earlier_prices(synthetic_market):
    """The other half of the cutoff: history before the session must still matter."""
    frame = synthetic_market(140, seed=4)
    features, _ = build_market_features(frame, moving_average_windows=(10, 50))
    feature_columns = [column for column in features.columns if column not in (*CONTEXT_COLUMNS, *TARGET_COLUMNS)]
    session = str(features["session_date"].iloc[80])
    before = features.set_index("session_date").loc[session]

    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    mutated = ordered.copy()
    mutated.loc[0:10, ["open", "high", "low", "close"]] = mutated.loc[0:10, ["open", "high", "low", "close"]] * 0.5

    after_features, _ = build_market_features(mutated, moving_average_windows=(10, 50))
    after = after_features.set_index("session_date").loc[session]
    changed = [column for column in feature_columns if not np.isclose(after[column], before[column])]
    assert changed, "features ignored the whole history, which would mean they are constant"


def test_series_shorter_than_the_warmup_is_refused():
    with pytest.raises(DataSourceError, match="warm-up"):
        build_market_features(_market([100.0] * 5), moving_average_windows=(10,))
