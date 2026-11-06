from __future__ import annotations

from typing import Any

from abc_sim.spelling_goal import SpellingGoal

from ..core import PerturbRange, RandomizationState, ScalePerturbRange, SceneRandomizer


class BlocksRandomizer(SceneRandomizer):
    """Per-block perturbation for the 26-letter blocks scene.

    Blocks are arranged in a 4-row grid with 55 mm spacing; perturbations
    scatter blocks across the table (±80 mm x, ±450 mm y, ±0.25 rad yaw).
    """

    min_clearance_m = 0.01
    # Dense 26-block placements need a larger retry budget.
    max_tries = 2000
    perturbations = [
        PerturbRange(f"block_{letter}_jnt", delta_x=(-0.08, 0.08), delta_y=(-0.45, 0.45), delta_yaw=(-0.25, 0.25))
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        return []

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        """Sample block poses and an explicitly requested spelling sequence."""
        if request is None:
            return super().randomize(model, data, seed=seed, request=request)
        if not isinstance(request, dict):
            raise TypeError("A blocks reset request must be a dict.")
        unknown = set(request) - {"sequence_length", "randomize_scales"}
        if unknown:
            raise ValueError(f"Unknown blocks reset options: {sorted(unknown)}")
        if "sequence_length" not in request:
            return super().randomize(model, data, seed=seed, request=request)
        goal = SpellingGoal.sample(request["sequence_length"], seed)
        state = super().randomize(
            model,
            data,
            seed=seed,
            request={key: value for key, value in request.items() if key != "sequence_length"},
        )
        state.metadata.update(goal.to_metadata())
        return state


__all__ = ["BlocksRandomizer"]
