"""Configured spelling goals survive batched metric discovery and world records."""

import json
import unittest
from unittest.mock import patch

import numpy as np
from test_spelling_goals import block_model, goal_state, mujoco, row_state

if mujoco is not None:
    from abc_sim.batched_env import PerWorldEvaluators
    from abc_sim.spelling_goal import SpellingGoal
    from abc_sim.task_eval.spell import SpellWordEvaluator
    from abc_sim.task_specs import get_task_spec


@unittest.skipIf(mujoco is None, "MuJoCo is required for native batch composition.")
class BatchedSpellingGoalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = block_model()

    def rows(self, words):
        return np.concatenate([row_state(self.model, word) for word in words])

    def test_configure_discover_evaluate_preserves_every_goal(self):
        for goals in (("AB", "XY"), ("ABC", "XYZ"), ("ABCD", "WXYZ")):
            for width in (1, 2):
                with self.subTest(goals=goals, width=width):
                    selected = goals[:width]
                    wrapper = PerWorldEvaluators(
                        self.model,
                        get_task_spec(f"spell_random_{len(goals[0])}"),
                        width,
                    )
                    wrapper.reset(self.model, [goal_state(word) for word in selected])
                    rows = self.rows(selected)
                    keys = wrapper.per_world_keys(rows)
                    self.assertIn("word", keys)
                    result = wrapper.evaluate_qpos_batch(rows)
                    np.testing.assert_array_equal(result.success, [True] * width)
                    self.assertEqual(result.metrics["word"], list(selected))

    def test_discovery_neither_scores_nor_resets_live_episode_state(self):
        wrapper = PerWorldEvaluators(self.model, get_task_spec("spell_random_3"), 1)
        wrapper.reset(self.model, [goal_state("ABC")])
        row = self.rows(["ABC"])
        live = wrapper.evaluators[0]
        with (
            patch.object(live, "reset", side_effect=AssertionError("live reset")),
            patch.object(
                live, "evaluate_qpos_batch", side_effect=AssertionError("live score")
            ),
        ):
            wrapper.per_world_keys(row)
        failed = wrapper.evaluate_qpos_batch(self.rows(["XBC"]))
        np.testing.assert_array_equal(failed.success, [False])
        np.testing.assert_array_equal(failed.metrics["ever_success"], [False])
        wrapper.evaluate_qpos_batch(row)
        wrapper.per_world_keys(self.rows(["XBC"]))
        failed = wrapper.evaluate_qpos_batch(self.rows(["XBC"]))
        np.testing.assert_array_equal(failed.metrics["ever_success"], [True])

    def test_second_reset_rebinds_goals_and_clears_each_world_history(self):
        wrapper = PerWorldEvaluators(self.model, get_task_spec("spell_random_3"), 2)
        first = ["ABC", "XYZ"]
        rows = self.rows(first)
        wrapper.reset(self.model, [goal_state(word) for word in first])
        wrapper.per_world_keys(rows)
        np.testing.assert_array_equal(
            wrapper.evaluate_qpos_batch(rows).success, [True, True]
        )

        second = ["XYZ", "ABC"]
        wrapper.reset(self.model, [goal_state(word, seed=38) for word in second])
        wrapper.per_world_keys(rows)
        wrong = wrapper.evaluate_qpos_batch(rows)
        self.assertEqual(wrong.metrics["word"], second)
        np.testing.assert_array_equal(wrong.success, [False, False])
        np.testing.assert_array_equal(wrong.metrics["ever_success"], [False, False])
        np.testing.assert_array_equal(
            wrapper.evaluate_qpos_batch(self.rows(second)).success, [True, True]
        )

    def test_world_records_keep_distinct_goals_rewards_and_metadata(self):
        from abc_minimal.batched_eval import world_eval

        wrapper = PerWorldEvaluators(self.model, get_task_spec("spell_random_3"), 2)
        for goals, seed in ((["ABC", "XYZ"], 37), (["XYZ", "ABC"], 38)):
            with self.subTest(goals=goals):
                states = [goal_state(word, seed=seed) for word in goals]
                wrapper.reset(self.model, states)
                rows = self.rows(goals)
                keys = wrapper.per_world_keys(rows)
                result = wrapper.evaluate_qpos_batch(rows)
                records = [
                    json.loads(json.dumps(world_eval(result, index, keys)))
                    for index in range(2)
                ]
                for index, record in enumerate(records):
                    goal = SpellingGoal.from_metadata(states[index].metadata)
                    self.assertEqual(record["word"], goal.sequence)
                    self.assertIn(
                        record["word"].lower(), states[index].metadata["prompt"]
                    )
                    self.assertEqual(record["reward"], 1.0)
                    self.assertTrue(record["success"])

    def test_fixed_word_keeps_single_world_serialization_and_native_rewards(self):
        from abc_minimal.batched_eval import world_eval

        spec = get_task_spec("spell_abc")
        wrapper = PerWorldEvaluators(self.model, spec, 2)
        wrapper.reset(self.model, [None, None])
        rows = self.rows(["ABC", "XBC"])
        keys = wrapper.per_world_keys(rows)
        result = wrapper.evaluate_qpos_batch(rows)
        np.testing.assert_allclose(result.reward, [1.0, 0.8])
        np.testing.assert_array_equal(result.success, [True, False])
        self.assertEqual(result.metrics["word"], ["ABC", "ABC"])
        direct = SpellWordEvaluator(model=self.model, spec=spec, word="abc")
        for index in range(2):
            direct.reset(nworld=1)
            expected = direct.evaluate_qpos_batch(rows[index : index + 1]).to_info(
                squeeze=True
            )
            actual = world_eval(result, index, keys)
            self.assertEqual(actual, expected)
            self.assertEqual(actual["word"], "ABC")

    def test_real_reset_still_requires_explicit_goal_rebinding(self):
        wrapper = PerWorldEvaluators(self.model, get_task_spec("spell_random_3"), 1)
        wrapper.reset(self.model, [goal_state("ABC")])
        rows = self.rows(["ABC"])
        wrapper.per_world_keys(rows)
        self.assertTrue(wrapper.evaluate_qpos_batch(rows).success[0])
        wrapper.evaluators[0].reset(nworld=1)
        with self.assertRaisesRegex(RuntimeError, "after every reset"):
            wrapper.evaluate_qpos_batch(rows)


if __name__ == "__main__":
    unittest.main()
