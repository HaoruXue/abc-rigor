"""Shared early-stop rules for sequential and batched policy rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from abc_sim.task_eval.base import TaskEvalResult


def _terminal_flags(metrics: Mapping[str, Any]) -> np.ndarray:
    """Completion ends on success or failure; maintenance ends only on failure."""
    mode = metrics.get("termination_mode", "completion")
    if mode == "completion":
        flags = np.logical_or(metrics["ever_success"], metrics.get("ever_failed", False))
    elif mode == "maintenance":
        flags = metrics["ever_failed"]
    else:
        raise ValueError(f"Unknown rollout termination_mode: {mode!r}")
    return np.asarray(flags, dtype=bool)


def rollout_over(task_eval: Mapping[str, Any]) -> bool:
    """Whether a single world can stop before exhausting its action budget."""
    return bool(_terminal_flags(task_eval))


def batched_rollout_over(result: TaskEvalResult | None, num_worlds: int) -> np.ndarray:
    """Return one terminal flag per world, using the sequential rollout's rules."""
    if result is None:
        return np.zeros(num_worlds, dtype=bool)
    return _terminal_flags(result.metrics)
