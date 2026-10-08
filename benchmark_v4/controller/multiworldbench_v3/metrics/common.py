"""Shared validation and aggregation helpers for V3 metrics."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Iterable, Mapping


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def score_0_100(value: Any, label: str) -> float:
    result = finite_number(value, label)
    if not 0.0 <= result <= 100.0:
        raise ValueError(f"{label} must be in [0, 100]")
    return result


def unit_interval(value: Any, label: str) -> float:
    result = finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def binary(value: Any, label: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{label} must be integer 0 or 1")
    return value


def arithmetic_mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise ValueError("at least one value is required")
    return float(fmean(rows))


def geometric_mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise ValueError("at least one value is required")
    if any(value < 0 for value in rows):
        raise ValueError("geometric mean inputs must be non-negative")
    if any(value == 0 for value in rows):
        return 0.0
    return math.exp(fmean(math.log(value) for value in rows))


def majority_vote(values: Iterable[int]) -> int:
    rows = [binary(value, "vote") for value in values]
    if not rows:
        raise ValueError("at least one vote is required")
    return int(sum(rows) >= (len(rows) // 2 + 1))


def aggregate_numeric_cases(
    rows: Iterable[Mapping[str, Any]], *, score_key: str = "score"
) -> dict[str, Any]:
    rows = list(rows)
    eligible = [score_0_100(row[score_key], score_key) for row in rows
                if row.get(score_key) is not None]
    return {
        "score": arithmetic_mean(eligible) if eligible else None,
        "eligible_cases": len(eligible),
        "total_cases": len(rows),
        "coverage": len(eligible) / len(rows) if rows else 0.0,
    }
