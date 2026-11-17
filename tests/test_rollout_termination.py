"""CPU regressions for completion and maintenance rollout termination."""

from __future__ import annotations

import json
import unittest

import numpy as np

from abc_minimal.termination import batched_rollout_over, rollout_over
from abc_sim import make_env
from abc_sim.batched_env import PerWorldEvaluators
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.count_box import CountIntoOpaqueBoxEvaluator


class RolloutTerminationTest(unittest.TestCase):
    def result(self, *, mode=None, include_failure=True):
        metrics = {"ever_success": np.array([False, True, False, True])}
        if include_failure:
            metrics["ever_failed"] = np.array([False, False, True, True])
        if mode is not None:
            metrics["termination_mode"] = mode
        return TaskEvalResult(
            reward=np.zeros(4),
            success=np.zeros(4, dtype=bool),
            metrics=metrics,
        )

    def assert_terminals(self, result, expected):
        np.testing.assert_array_equal(batched_rollout_over(result, 4), expected)
        for index, terminal in enumerate(expected):
            single = TaskEvalResult(
                reward=result.reward[index : index + 1],
                success=result.success[index : index + 1],
                metrics={
                    key: value[index : index + 1] if isinstance(value, np.ndarray) else value
                    for key, value in result.metrics.items()
                },
            )
            info = json.loads(json.dumps(single.to_info(squeeze=True)))
            with self.subTest(index=index, info=info):
                self.assertEqual(rollout_over(info), terminal)

    def test_completion_stops_for_success_failure_or_both(self):
        self.assert_terminals(self.result(), [False, True, True, True])
        self.assert_terminals(self.result(mode="completion"), [False, True, True, True])

    def test_completion_without_failure_metric_stops_on_latched_success(self):
        self.assert_terminals(self.result(include_failure=False), [False, True, False, True])

    def test_maintenance_continues_success_and_stops_only_for_failure(self):
        self.assert_terminals(self.result(mode="maintenance"), [False, False, True, True])

    def test_unknown_mode_fails_in_both_consumers(self):
        result = self.result(mode="maintenence")
        with self.assertRaisesRegex(ValueError, "termination_mode"):
            batched_rollout_over(result, 4)
        with self.assertRaisesRegex(ValueError, "termination_mode"):
            rollout_over({"ever_success": False, "termination_mode": "maintenence"})

    def test_missing_evaluator_keeps_every_world_running(self):
        np.testing.assert_array_equal(batched_rollout_over(None, 3), [False] * 3)
        self.assertFalse(rollout_over({"reward": 0.0, "success": False, "ever_success": False}))


class CountBoxTerminationTest(unittest.TestCase):
    def setUp(self):
        self.env = make_env("count_three_into_opaque_box", render_cameras=False)
        self.addCleanup(self.env.close)
        self.env.reset(seed=0)

    def qpos_with_count(self, count):
        evaluator = self.env._task_evaluator
        qpos = self.env.data.qpos.copy()
        x, y = np.mean(evaluator._box_x_bounds), np.mean(evaluator._box_y_bounds)
        for name in evaluator._directive["names"][:count]:
            adr = self.env.model.joint(name + "_joint").qposadr[0]
            qpos[adr : adr + 3] = (x, y, evaluator._box_floor_z + 0.02)
        return qpos

    def test_correct_count_can_become_incorrect_without_early_stop(self):
        evaluator = self.env._task_evaluator
        for count, success in ((2, False), (3, True), (4, False)):
            with self.subTest(count=count):
                result = evaluator.evaluate_qpos_batch(self.qpos_with_count(count)[None])
                info = json.loads(json.dumps(result.to_info(squeeze=True)))
                self.assertEqual(info["termination_mode"], "maintenance")
                self.assertEqual(info["success"], success)
                self.assertEqual(info["ever_success"], success)
                self.assertFalse(rollout_over(info))
                np.testing.assert_array_equal(batched_rollout_over(result, 1), [False])
        self.assertTrue(info["ever_correct"])

    def test_unconfigured_count_still_declares_maintenance(self):
        evaluator = CountIntoOpaqueBoxEvaluator(model=self.env.model, spec=self.env._task_spec)
        result = evaluator.evaluate_qpos_batch(self.env.data.qpos[None])
        self.assertFalse(result.metrics["goal_configured"])
        self.assertEqual(result.metrics["termination_mode"], "maintenance")
        self.assertFalse(rollout_over(result.to_info(squeeze=True)))

    def test_per_world_aggregation_preserves_maintenance_and_world_axes(self):
        evaluators = PerWorldEvaluators(self.env.model, self.env._task_spec, 2)
        evaluators.reset(self.env.model, [self.env._last_randomization] * 2)
        qpos = np.stack([self.qpos_with_count(3), self.qpos_with_count(4)])
        keys = evaluators.per_world_keys(qpos)
        self.assertNotIn("termination_mode", keys)
        self.assertIn("ever_success", keys)
        result = evaluators.evaluate_qpos_batch(qpos)
        info = json.loads(json.dumps(result.to_info()))
        self.assertEqual(info["termination_mode"], "maintenance")
        self.assertEqual(info["success"], [True, False])
        np.testing.assert_array_equal(batched_rollout_over(result, 2), [False, False])
        for index in range(2):
            world = {key: value[index] if key in keys | {"reward", "success"} else value for key, value in info.items()}
            self.assertFalse(rollout_over(world))


if __name__ == "__main__":
    unittest.main()
