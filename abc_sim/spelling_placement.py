"""Conservative spelling geometry and measured support, revision v1."""

from __future__ import annotations

import math
import string
from dataclasses import dataclass, fields
from itertools import product
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FreePlacementSpellingCfg:
    """Versioned geometry and continuous physical eligibility thresholds."""

    revision: str = "abc-spelling-free-placement-v1"
    row_width_m: float = 0.03
    gap_min_m: float = 0.025
    gap_max_m: float = 0.06
    displacement_m: float = 0.02
    upright_dot_min: float = 0.85
    minimum_weight_fraction: float = 0.5
    max_linear_speed: float = 0.01
    max_angular_speed: float = 0.1
    settling_seconds: float = 0.1
    robot_force_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.revision != "abc-spelling-free-placement-v1":
            raise ValueError("Unsupported free-placement spelling revision.")
        for field in fields(self):
            if field.name == "revision":
                continue
            value = getattr(self, field.name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name} must be positive and finite.")
        if self.upright_dot_min > 1 or self.minimum_weight_fraction > 1:
            raise ValueError(
                "Spelling alignment and weight fractions cannot exceed one."
            )
        if self.gap_min_m > self.gap_max_m:
            raise ValueError("Spelling spacing bounds must be ordered.")


def score_free_placement(
    cfg: FreePlacementSpellingCfg,
    *,
    word: str,
    letters: list[str],
    positions: list[list[float]],
    initial_positions: list[list[float]],
    face_up: list[bool],
    supported: list[bool],
) -> dict[str, Any]:
    """Credit placements correct under every geometry-admissible interpretation.

    Candidate components contain at least two projected blocks and at least one
    block displaced from reset. All occupants participate, including wrong,
    unreadable and held blocks. Reading proceeds toward decreasing scene Y.
    Geometric gaps admit missing positions. Every fitting endpoint offset is
    considered without consulting letter identity. Separate candidate rows are
    competing interpretations, so only placements common to all earn credit.
    """
    count = len(letters)
    if (
        not isinstance(word, str)
        or not 2 <= len(word) <= 4
        or len(set(word)) != len(word)
        or not set(word) <= set(string.ascii_uppercase)
        or len(set(letters)) != count
        or any(
            not isinstance(letter, str)
            or len(letter) != 1
            or letter not in string.ascii_uppercase
            for letter in letters
        )
        or not set(word) <= set(letters)
    ):
        raise ValueError(
            "Spelling requires a distinct uppercase target and physical letter inventory."
        )
    current, initial = (
        np.asarray(positions, dtype=np.float64),
        np.asarray(initial_positions, dtype=np.float64),
    )
    if (
        current.shape != (count, 3)
        or initial.shape != current.shape
        or not np.isfinite(current).all()
        or not np.isfinite(initial).all()
        or any(
            len(flags) != count or any(type(flag) is not bool for flag in flags)
            for flags in (face_up, supported)
        )
    ):
        raise ValueError(
            "Spelling requires complete finite positions and explicit boolean eligibility."
        )
    moved = (
        np.linalg.norm(current[:, :2] - initial[:, :2], axis=1) >= cfg.displacement_m
    )
    delta = np.abs(current[:, None, :2] - current[None, :, :2])
    adjacent = (delta[..., 0] <= cfg.row_width_m) & (
        delta[..., 1] <= (len(word) - 1) * cfg.gap_max_m
    )
    remaining = set(range(count))
    rows = []
    while remaining:
        component, frontier = set(), {min(remaining)}
        while frontier:
            index = frontier.pop()
            component.add(index)
            remaining.remove(index)
            frontier.update(
                int(other)
                for other in np.flatnonzero(adjacent[index])
                if other in remaining
            )
            frontier.difference_update(component)
        indices = sorted(component, key=lambda index: (-current[index, 1], index))
        if len(indices) >= 2 and moved[indices].any():
            rows.append(indices)

    common: set[int] | None = None
    interpretations = 0
    denominator = len(word)
    diagnostics = []
    for indices in rows:
        gaps = -np.diff(current[indices, 1])
        # Positive unit jumps enforce one-to-one positions, including collisions.
        max_span = max(len(word), len(indices))
        jump_options = [
            [
                jump
                for jump in range(1, max_span - len(indices) + 2)
                if cfg.gap_min_m <= gap / jump <= cfg.gap_max_m
            ]
            for gap in gaps
        ]
        row_interpretations = 0
        if np.ptp(current[indices, 0]) <= cfg.row_width_m:
            for jumps in product(*jump_options):
                span = 1 + sum(jumps)
                if span > max_span:
                    continue
                slots = np.concatenate(([0], np.cumsum(jumps)))
                offsets = range(min(0, len(word) - span), max(0, len(word) - span) + 1)
                for offset in offsets:
                    correct = {
                        index
                        for index, slot in zip(indices, slots + offset, strict=True)
                        if 0 <= slot < len(word)
                        and letters[index] == word[slot]
                        and face_up[index]
                        and supported[index]
                    }
                    common = correct if common is None else common & correct
                    row_interpretations += 1
        if row_interpretations:
            denominator = max(denominator, len(indices))
        interpretations += row_interpretations
        diagnostics.append(
            {
                "occupants": [letters[index] for index in indices],
                "interpretations": row_interpretations,
            }
        )
    credited = [] if common is None else sorted(letters[index] for index in common)
    return {
        "revision": cfg.revision,
        "credited_letters": credited,
        "correct_placements": len(credited),
        "reward_denominator": denominator,
        "reward": len(credited) / denominator,
        "success": len(credited) == len(word) and denominator == len(word),
        "interpretations": interpretations,
        "rows": diagnostics,
    }


class SpellingTableSupport:
    """Track direct table support and loads from explicitly supplied geoms."""

    def __init__(
        self,
        cfg: FreePlacementSpellingCfg,
        model: Any,
        *,
        body_names: list[str],
        robot_geom_ids: set[int],
    ) -> None:
        import mujoco

        self.cfg, self.model = cfg, model
        self.body_ids = np.asarray([model.body(name).id for name in body_names])
        if not len(self.body_ids) or len(set(self.body_ids)) != len(self.body_ids):
            raise ValueError("Support bodies must be nonempty and unique.")
        if not robot_geom_ids:
            raise ValueError(
                "Spelling support requires explicitly identified robot geoms."
            )
        self.robot_geom_ids = set(robot_geom_ids)
        self.body_to_index = {
            int(body): index for index, body in enumerate(self.body_ids)
        }
        self.table_id = model.geom("table_plane").id
        if model.geom_type[self.table_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("Spelling support requires a planar table.")
        joints = model.body_jntadr[self.body_ids]
        if np.any(model.body_jntnum[self.body_ids] != 1) or np.any(
            model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError("Spelling blocks require one free joint each.")
        self.dof_addresses = model.jnt_dofadr[joints]
        gravity = np.asarray(model.opt.gravity)
        gravity_magnitude = np.linalg.norm(gravity)
        self.weights = model.body_mass[self.body_ids] * gravity_magnitude
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0):
            raise ValueError("Spelling blocks require positive finite weight.")
        self.up = -gravity / gravity_magnitude
        self.sustained = np.zeros(len(self.body_ids))
        self.previous_eligible = np.zeros(len(self.body_ids), dtype=bool)
        self.previous_time: float | None = None

    def sample(self, data: Any) -> dict[str, list]:
        """Read solved contacts and require an uninterrupted settling interval."""
        import mujoco

        now = float(data.time)
        elapsed = 0.0 if self.previous_time is None else now - self.previous_time
        if not np.isfinite(now) or elapsed < 0:
            raise ValueError(
                "Support samples require finite chronological physics time."
            )
        if self.previous_time is not None and not np.isclose(
            elapsed, self.model.opt.timestep, rtol=1e-7, atol=1e-12
        ):
            raise ValueError("Spelling support must sample every native physics step.")
        self.previous_time = now
        normal = np.asarray(data.geom_xmat[self.table_id]).reshape(3, 3)[:, 2]
        if not np.isclose(abs(np.dot(normal, self.up)), 1, atol=1e-6, rtol=0):
            raise ValueError(
                "The spelling table must be horizontal relative to gravity."
            )
        table_load, robot_load, force = (
            np.zeros(len(self.body_ids)),
            np.zeros(len(self.body_ids)),
            np.zeros(6),
        )
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            mujoco.mj_contactForce(self.model, data, contact_index, force)
            if not np.isfinite(force).all():
                raise ValueError("Spelling contact forces must be finite.")
            for target_geom, other_geom in (
                (contact.geom1, contact.geom2),
                (contact.geom2, contact.geom1),
            ):
                body = int(self.model.geom_bodyid[target_geom])
                if body not in self.body_to_index:
                    continue
                index = self.body_to_index[body]
                if other_geom == self.table_id:
                    contact_normal = np.asarray(contact.frame).reshape(3, 3)[0]
                    table_load[index] += force[0] * abs(np.dot(contact_normal, self.up))
                elif other_geom in self.robot_geom_ids:
                    robot_load[index] += force[0]
        velocity = np.asarray(
            [data.qvel[address : address + 6] for address in self.dof_addresses]
        )
        if not np.isfinite(velocity).all():
            raise ValueError("Spelling velocities must be finite.")
        linear, angular = (
            np.linalg.norm(velocity[:, :3], axis=1),
            np.linalg.norm(velocity[:, 3:], axis=1),
        )
        weight_fraction = table_load / self.weights
        eligible = (
            (weight_fraction >= self.cfg.minimum_weight_fraction)
            & (robot_load <= self.cfg.robot_force_tolerance)
            & (linear <= self.cfg.max_linear_speed)
            & (angular <= self.cfg.max_angular_speed)
        )
        self.sustained = np.where(
            eligible & self.previous_eligible, self.sustained + elapsed, 0.0
        )
        self.previous_eligible = eligible.copy()
        return {
            "eligible": (self.sustained >= self.cfg.settling_seconds).tolist(),
            "sustained_seconds": self.sustained.tolist(),
            "weight_fraction": weight_fraction.tolist(),
            "robot_normal_load": robot_load.tolist(),
            "linear_speed": linear.tolist(),
            "angular_speed": angular.tolist(),
        }
