"""Data-quality reporting.

Every cleaning step returns one of these alongside the cleaned table, and the
Phase 1 artifact embeds the JSON form. The point is that missing values,
duplicates, and timezone assumptions are *reported* numbers, never silent
decisions (see AGENTS.md §4 invariant 3 and §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd


def missing_counts(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> dict[str, int]:
    """Count missing (NaN/None/empty-string) values per column."""
    cols = list(columns) if columns is not None else list(frame.columns)
    counts: dict[str, int] = {}
    for column in cols:
        if column not in frame.columns:
            counts[column] = 0
            continue
        series = frame[column]
        missing = series.isna()
        if series.dtype == object:
            missing = missing | series.astype("string").fillna("").str.strip().eq("")
        counts[column] = int(missing.sum())
    return counts


def _empty_string_missing(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.isna() | series.astype("string").fillna("").str.strip().eq("")
    return series.isna()


def missing_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    """Boolean mask of missing values in ``column`` (blank strings count as missing)."""
    if column not in frame.columns:
        return pd.Series(True, index=frame.index)
    return _empty_string_missing(frame[column])


@dataclass
class QualityReport:
    """What a cleaning step changed, and what it refused to guess."""

    table: str
    rows_read: int = 0
    rows_written: int = 0
    duplicate_key: str = ""
    duplicates_removed: int = 0
    rows_dropped_unparsable_timestamp: int = 0
    rows_dropped_missing_required: int = 0
    rows_outside_window: int = 0
    coerced_to_missing: dict[str, int] = field(default_factory=dict)
    missing_counts: dict[str, int] = field(default_factory=dict)
    missing_ratio: dict[str, float] = field(default_factory=dict)
    input_chronological: bool | None = None
    date_range: tuple[str, str] | None = None
    timezone_assumptions: list[str] = field(default_factory=list)
    imputation_applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # -- mutators -----------------------------------------------------------------
    def record_missing(self, frame: pd.DataFrame, columns: Iterable[str] | None = None) -> None:
        counts = missing_counts(frame, columns)
        self.missing_counts.update(counts)
        total = int(len(frame))
        for column, count in counts.items():
            self.missing_ratio[column] = round(count / total, 6) if total else 0.0

    def record_date_range(self, series: pd.Series) -> None:
        valid = series.dropna()
        if valid.empty:
            self.date_range = None
            return
        self.date_range = (_iso(valid.min()), _iso(valid.max()))

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def add_timezone_assumption(self, message: str) -> None:
        if message not in self.timezone_assumptions:
            self.timezone_assumptions.append(message)

    def add_note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    def record_imputation(self, column: str) -> None:
        """Register any fill/imputation. Project rule: absence of news is never imputed."""
        if column not in self.imputation_applied:
            self.imputation_applied.append(column)

    def columns_with_missing(self) -> dict[str, int]:
        return {column: count for column, count in self.missing_counts.items() if count}

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "duplicate_key": self.duplicate_key,
            "duplicates_removed": self.duplicates_removed,
            "rows_dropped_unparsable_timestamp": self.rows_dropped_unparsable_timestamp,
            "rows_dropped_missing_required": self.rows_dropped_missing_required,
            "rows_outside_window": self.rows_outside_window,
            "coerced_to_missing": dict(sorted(self.coerced_to_missing.items())),
            "missing_counts": dict(sorted(self.missing_counts.items())),
            "missing_ratio": dict(sorted(self.missing_ratio.items())),
            "columns_with_missing": dict(sorted(self.columns_with_missing().items())),
            "input_chronological": self.input_chronological,
            "date_range": list(self.date_range) if self.date_range else None,
            "timezone_assumptions": list(self.timezone_assumptions),
            "imputation_applied": list(self.imputation_applied),
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }


def _iso(value: Any) -> str:
    stamp = pd.Timestamp(value)
    return stamp.isoformat().replace("+00:00", "Z")


__all__ = ["QualityReport", "missing_counts", "missing_mask"]
