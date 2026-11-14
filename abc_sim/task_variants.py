"""Versioned short-task definitions using the existing native scenes and scores."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from abc_sim.task_specs import SimTaskSpec, get_task_spec


def _count(value: int, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"Expected an integer count in [{lower}, {upper}].")


def _horizon(value: int) -> None:
    if type(value) is not int or value not in (30, 60):
        raise ValueError("Throughput v1 horizons are exactly 30 or 60 seconds.")


def _variant(
    base: str,
    name: str,
    horizon: int,
    reset: tuple[tuple[str, object], ...],
    description: str,
) -> SimTaskSpec:
    return replace(
        get_task_spec(base),
        name=name,
        aliases=(),
        version="v1",
        base_task=base,
        horizon_seconds=horizon,
        reset_request=reset,
        description=description,
    )


@dataclass(frozen=True)
class BottleThroughputTask:
    """A fixed active bottle count and a versioned simulated-time budget."""

    bottle_count: int = 6
    horizon_seconds: int = 30

    def __post_init__(self) -> None:
        _count(self.bottle_count, 2, 6)
        _horizon(self.horizon_seconds)

    @property
    def task_id(self) -> str:
        return f"throughput_bottles{self.bottle_count}_{self.horizon_seconds}s_v1"

    @property
    def active_object_count(self) -> int:
        return self.bottle_count

    def task_spec(self) -> SimTaskSpec:
        return _variant(
            "put_plastic_bottles_in_bin",
            self.task_id,
            self.horizon_seconds,
            (("bottle_count", self.bottle_count),),
            "Versioned fixed-count, short-horizon bottle training condition.",
        )

    def validate_reset(self, randomization: dict[str, Any]) -> None:
        if randomization["metadata"]["bottle_count"] != self.bottle_count:
            raise ValueError(
                "The bottle reset did not honor the declared active count."
            )

    def progress_counts(
        self, metrics: dict[str, Any]
    ) -> tuple[int, int, dict[str, int]]:
        return metrics["num_bottles_in_bin"], metrics["num_active_bottles"], {}


@dataclass(frozen=True)
class LegoThroughputTask:
    """A fixed number of each native LEGO color and a simulated-time budget."""

    blocks_per_color: int = 2
    horizon_seconds: int = 30

    def __post_init__(self) -> None:
        _count(self.blocks_per_color, 1, 9)
        _horizon(self.horizon_seconds)

    @property
    def task_id(self) -> str:
        return (
            f"throughput_lego{self.blocks_per_color}percolor_{self.horizon_seconds}s_v1"
        )

    @property
    def active_object_count(self) -> int:
        return 3 * self.blocks_per_color

    def task_spec(self) -> SimTaskSpec:
        return _variant(
            "sort_lego_blocks",
            self.task_id,
            self.horizon_seconds,
            (("blocks_per_color", self.blocks_per_color),),
            "Versioned fixed-count, short-horizon LEGO training condition.",
        )

    def validate_reset(self, randomization: dict[str, Any]) -> None:
        expected = {color: self.blocks_per_color for color in ("red", "yellow", "blue")}
        if randomization["metadata"]["active_counts"] != expected:
            raise ValueError("The LEGO reset did not honor every declared color count.")

    def progress_counts(
        self, metrics: dict[str, Any]
    ) -> tuple[int, int, dict[str, int]]:
        return metrics["num_correct"], metrics["active_object_count"], {}


@dataclass(frozen=True)
class NutsBoltsThroughputTask:
    """Fixed native nut and bolt counts and a simulated-time budget."""

    nut_count: int = 3
    bolt_count: int = 3
    horizon_seconds: int = 30

    def __post_init__(self) -> None:
        _count(self.nut_count, 1, 20)
        _count(self.bolt_count, 1, 20)
        _horizon(self.horizon_seconds)

    @property
    def task_id(self) -> str:
        return f"throughput_nuts{self.nut_count}_bolts{self.bolt_count}_{self.horizon_seconds}s_v1"

    @property
    def active_object_count(self) -> int:
        return self.nut_count + self.bolt_count

    def task_spec(self) -> SimTaskSpec:
        return _variant(
            "sort_nuts_and_bolts",
            self.task_id,
            self.horizon_seconds,
            (("nut_count", self.nut_count), ("bolt_count", self.bolt_count)),
            "Versioned fixed-count, short-horizon nuts-and-bolts training condition.",
        )

    def validate_reset(self, randomization: dict[str, Any]) -> None:
        expected = {"nuts": self.nut_count, "bolts": self.bolt_count}
        if randomization["metadata"]["active_counts"] != expected:
            raise ValueError(
                "The sorting reset did not honor the declared nuts and bolts counts."
            )

    def progress_counts(
        self, metrics: dict[str, Any]
    ) -> tuple[int, int, dict[str, int]]:
        return (
            metrics["num_correct"],
            metrics["active_object_count"],
            {
                "num_correct_spec": metrics["num_correct_spec"],
                "num_correct_inverted": metrics["num_correct_inverted"],
            },
        )


def free_placement_spelling_task() -> SimTaskSpec:
    """Define the shipped two-letter, 60-second conservative spelling task."""
    base = get_task_spec("spell_random_2")
    return replace(
        base,
        name="spell_random_2_free_placement_60s_v1",
        version="v1",
        base_task=base.name,
        horizon_seconds=60,
        evaluator_name="abc-spelling-free-placement-v1",
        evaluator_options=(("sequence_length", 2), ("evaluation_interval_steps", 17)),
        aliases=(),
        description="Seeded two-letter spelling with conservative placement and measured table support.",
    )
