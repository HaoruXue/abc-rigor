"""Native contact-dependent evaluator for versioned free-placement spelling."""

from __future__ import annotations

import string
from copy import deepcopy
from dataclasses import asdict

import mujoco
import numpy as np

from abc_sim.spelling_placement import (
    FreePlacementSpellingCfg,
    SpellingTableSupport,
    score_free_placement,
)
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.dishrack import _gripper_geom_ids
from abc_sim.task_eval.spell import SpellWordEvaluator
from abc_sim.task_specs import SimTaskSpec


class FreePlacementSpellingEvaluator:
    """Score resolved placement after uninterrupted native contact sampling.

    Qpos-only replay cannot supply contact loads or settling history. This
    evaluator therefore requires the single-world model/data physics hooks.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        sequence_length: int,
        evaluation_interval_steps: int = 17,
    ) -> None:
        if (
            type(evaluation_interval_steps) is not int
            or evaluation_interval_steps != 17
        ):
            raise ValueError(
                "Free-placement spelling v1 scores every 17 native physics steps."
            )
        self.model, self.spec = model, spec
        self.evaluation_interval_steps = evaluation_interval_steps
        self.cfg = FreePlacementSpellingCfg()
        self._native = SpellWordEvaluator(
            model=model, spec=spec, sequence_length=sequence_length
        )
        self._letters = list(string.ascii_uppercase)
        self.reset()

    def reset(self, *, nworld: int = 1) -> None:
        if nworld != 1:
            raise ValueError(
                "Free-placement spelling requires single-world contact observations."
            )
        self._native.reset(nworld=nworld)
        self.word = ""
        self._tracker = None
        self._data = None
        self._support = None
        self._ever_success = False
        self._physics_steps = 0
        self._result = None

    def configure_from_randomization(
        self, model: mujoco.MjModel, randomization: object
    ) -> None:
        self._native.configure_from_randomization(model, randomization)
        self.model = model
        self.word = self._native.word

    def after_reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if model is not self.model or not self.word:
            raise RuntimeError(
                "Bind the realized spelling goal before capturing reset geometry."
            )
        self._addresses = [
            int(model.joint(f"block_{letter}_jnt").qposadr[0])
            for letter in self._letters
        ]
        self._initial_positions = self._positions(data)
        self._origin_time = float(data.time)
        self._data = data
        self._tracker = SpellingTableSupport(
            self.cfg,
            model,
            body_names=[f"block_{letter}" for letter in self._letters],
            robot_geom_ids=_gripper_geom_ids(model),
        )
        self._support = None
        self._physics_steps = 0
        self._result = None

    def _positions(self, data: mujoco.MjData) -> list[list[float]]:
        return [
            data.qpos[address : address + 3].tolist() for address in self._addresses
        ]

    def after_physics_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if model is not self.model or data is not self._data or self._tracker is None:
            raise RuntimeError(
                "Reset contact tracking before observing spelling physics."
            )
        self._support = self._tracker.sample(data)
        self._physics_steps += 1
        if self._physics_steps % self.evaluation_interval_steps == 0:
            self._result = self._score_model_data(data)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        raise NotImplementedError(
            "Free-placement spelling requires every native physics step and solved contacts."
        )

    def evaluate_model_data(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> TaskEvalResult:
        if model is not self.model or data is not self._data or self._result is None:
            raise RuntimeError(
                "Wait for the first scheduled spelling evaluation after reset."
            )
        return deepcopy(self._result)

    def _score_model_data(self, data: mujoco.MjData) -> TaskEvalResult:
        quat = np.asarray(
            [data.qpos[address + 3 : address + 7] for address in self._addresses]
        )
        local_x_up = 2 * (quat[:, 1] * quat[:, 3] - quat[:, 0] * quat[:, 2])
        frame = {
            "word": self.word,
            "letters": self._letters.copy(),
            "positions": self._positions(data),
            "initial_positions": [
                position.copy() for position in self._initial_positions
            ],
            "face_up": (np.abs(local_x_up) >= self.cfg.upright_dot_min).tolist(),
            "supported": self._support["eligible"].copy(),
        }
        score = score_free_placement(self.cfg, **frame)
        original = self._native.evaluate_qpos_batch(np.asarray(data.qpos)[None, :])
        self._ever_success |= score["success"]
        return TaskEvalResult(
            reward=np.asarray([score["reward"]]),
            success=np.asarray([score["success"]]),
            metrics={
                "word": self.word,
                "task_version": self.spec.version,
                "evaluator_revision": self.cfg.revision,
                "verifier_sample_count": self._physics_steps
                // self.evaluation_interval_steps,
                "verifier_time_s": float(data.time) - self._origin_time,
                "spelling_placement": {
                    "contact_scope": "finger-subtrees-v1",
                    "configuration": asdict(self.cfg),
                    "frame": frame,
                    "score": score,
                    "support": deepcopy(self._support),
                },
                "original_spell_reward": original.reward,
                "original_spell_success": original.success,
                "original_spell_ever_success": original.metrics["ever_success"],
                "ever_success": np.asarray([self._ever_success]),
            },
        )
