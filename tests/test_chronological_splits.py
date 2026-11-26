"""Chronological splits: contiguous ordered ranges, no shuffling, no gaps."""

from __future__ import annotations

import pytest

from stock_project.features.chronological_splits import chronological_split, train_validation_test
from stock_project.features.technical_indicators import build_market_features


def test_ranges_are_contiguous_and_cover_every_row_exactly_once():
    split = chronological_split(100, train_fraction=0.7, validation_fraction=0.15)
    assert split.train == range(0, 70)
    assert split.validation == range(70, 85)
    assert split.test == range(85, 100)
    covered = list(split.train) + list(split.validation) + list(split.test)
    assert covered == list(range(100))


def test_boundaries_floor_deterministically():
    split = chronological_split(101, train_fraction=0.7, validation_fraction=0.15)
    assert (len(split.train), len(split.validation), len(split.test)) == (70, 15, 16)
    assert split.as_dict() == {"train": [0, 70], "validation": [70, 85], "test": [85, 101]}


@pytest.mark.parametrize(
    "train_fraction, validation_fraction",
    [
        (0.0, 0.2),
        (1.0, 0.0),
        (0.8, 0.2),  # leaves no test period
        (0.9, 0.2),
        (0.5, -0.1),
    ],
)
def test_invalid_fractions_are_rejected(train_fraction, validation_fraction):
    with pytest.raises(ValueError):
        chronological_split(100, train_fraction=train_fraction, validation_fraction=validation_fraction)


def test_too_few_rows_are_rejected():
    with pytest.raises(ValueError):
        chronological_split(2, train_fraction=0.6, validation_fraction=0.2)


def test_periods_are_time_ordered_and_disjoint(synthetic_market):
    frame = synthetic_market(120)
    features, _ = build_market_features(frame, moving_average_windows=(10,))
    split = chronological_split(len(features), train_fraction=0.6, validation_fraction=0.2)
    periods = train_validation_test(features, split)

    assert periods["train"]["last_session"] < periods["validation"]["first_session"]
    assert periods["validation"]["last_session"] < periods["test"]["first_session"]
    assert (
        periods["train"]["rows"] + periods["validation"]["rows"] + periods["test"]["rows"] == len(features)
    )
    for name in ("train", "validation", "test"):
        period = periods[name]
        assert period["positive"] + period["negative"] == period["rows"]
        assert 0.0 <= period["positive_rate"] <= 1.0
