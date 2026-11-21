"""Chronological train/validation/test splitting.

The project never shuffles a time series (``docs/ARCHITECTURE.md`` §3, invariant
1). Rows are cut in order into contiguous periods: earlier rows train, the middle
selects hyper-parameters, and the final period is held out and touched only once,
after selection.

No purge gap is needed here, and that is a property worth stating: a feature row
for session ``t`` contains only information from strictly before ``t``, and its
target is session ``t``'s own move. Nothing in a row depends on another period, so
cutting between rows cannot leak anything across the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Context/target columns of a feature frame (kept for summaries).
DEFAULT_TIME_COLUMN = "session_date"
DEFAULT_TARGET_COLUMN = "target"


@dataclass(frozen=True)
class ChronologicalSplit:
    """Positional ranges of the three ordered periods."""

    train: range
    validation: range
    test: range

    def as_dict(self) -> dict[str, list[int]]:
        """``{period: [start, stop]}`` — JSON-serialisable, half-open ranges."""
        return {
            "train": [self.train.start, self.train.stop],
            "validation": [self.validation.start, self.validation.stop],
            "test": [self.test.start, self.test.stop],
        }


def chronological_split(
    row_count: int,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> ChronologicalSplit:
    """Cut ``row_count`` ordered rows into train / validation / test ranges.

    Fractions are applied with ``int()`` (floor) so the result is deterministic
    across platforms and floating-point noise cannot move a boundary. The test
    period is whatever remains, so the three ranges always cover every row
    exactly once, in order.
    """
    if row_count < 3:
        raise ValueError(f"need at least 3 rows to split chronologically, got {row_count}")
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0 < validation_fraction < 1:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must leave a non-empty test period, "
            f"got {train_fraction} + {validation_fraction}"
        )

    train_stop = int(row_count * train_fraction)
    validation_stop = train_stop + int(row_count * validation_fraction)
    # int() can floor to 0 for tiny inputs; refuse rather than hand back an empty period.
    if train_stop < 1 or validation_stop - train_stop < 1 or row_count - validation_stop < 1:
        raise ValueError(f"row_count={row_count} is too small for the configured split fractions")

    return ChronologicalSplit(
        train=range(0, train_stop),
        validation=range(train_stop, validation_stop),
        test=range(validation_stop, row_count),
    )


def slice_summary(
    frame: pd.DataFrame,
    rows: range,
    *,
    time_column: str = DEFAULT_TIME_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> dict[str, object]:
    """Rows, date coverage and class balance of one split (counts, not guesses)."""
    subset = frame.iloc[rows.start : rows.stop]
    positives = int((subset[target_column] == 1).sum())
    negatives = int((subset[target_column] == 0).sum())
    total = positives + negatives
    times = subset[time_column]
    return {
        "rows": int(len(subset)),
        "start_index": int(rows.start),
        "stop_index": int(rows.stop),
        "first_session": str(times.iloc[0]) if len(subset) else None,
        "last_session": str(times.iloc[-1]) if len(subset) else None,
        "positive": positives,
        "negative": negatives,
        "positive_rate": round(positives / total, 6) if total else None,
    }


def train_validation_test(
    frame: pd.DataFrame,
    split: ChronologicalSplit,
    *,
    time_column: str = DEFAULT_TIME_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> dict[str, dict[str, object]]:
    """Summaries for all three periods, keyed by period name."""
    return {
        "train": slice_summary(frame, split.train, time_column=time_column, target_column=target_column),
        "validation": slice_summary(frame, split.validation, time_column=time_column, target_column=target_column),
        "test": slice_summary(frame, split.test, time_column=time_column, target_column=target_column),
    }


__all__ = [
    "ChronologicalSplit",
    "DEFAULT_TARGET_COLUMN",
    "DEFAULT_TIME_COLUMN",
    "chronological_split",
    "slice_summary",
    "train_validation_test",
]
