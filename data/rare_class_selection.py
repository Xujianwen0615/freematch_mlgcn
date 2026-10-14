"""Select rare classes using only the labeled training split."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def parse_class_ids(value: Iterable[int] | str | None) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(index) for index in value]


def select_rare_classes(
    counts: Sequence[float],
    rare_class_ids: Iterable[int] | str | None = None,
    rare_quantile: float = 0.25,
) -> list[int]:
    """Return explicit IDs or positive-count classes below the chosen quantile."""
    values = np.asarray(counts, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("counts must be a nonempty sequence of finite nonnegative values")
    if not 0 <= rare_quantile <= 1:
        raise ValueError("rare_quantile must be in [0, 1]")
    explicit = sorted(set(parse_class_ids(rare_class_ids)))
    if explicit:
        if explicit[0] < 0 or explicit[-1] >= len(values):
            raise ValueError(f"rare class ID outside [0, {len(values) - 1}]")
        return explicit
    positive = values[values > 0]
    if not len(positive):
        return []
    threshold = np.quantile(positive, rare_quantile)
    return np.flatnonzero((values > 0) & (values <= threshold)).tolist()
