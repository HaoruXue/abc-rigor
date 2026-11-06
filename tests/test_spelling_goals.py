"""Exact spelling goals, reset binding, and fixed-word reward compatibility."""

import json
import string
import unittest
from copy import deepcopy
from unittest.mock import patch

import mujoco
import numpy as np
from abc_sim.randomization.core import RandomizationState, SceneRandomizer
from abc_sim.randomization.tasks.blocks import BlocksRandomizer
from abc_sim.spelling_goal import SpellingGoal
from abc_sim.task_eval.spell import SpellWordEvaluator
from abc_sim.task_specs import get_task_spec


def block_model():
    bodies = "".join(
        f'<body name="block_{letter}" pos="0.75 {0.02 * index} 0.765">'
        f'<freejoint name="block_{letter}_jnt"/>'
        '<geom type="box" size=".015 .015 .015" mass=".03"/></body>'
        for index, letter in enumerate(string.ascii_uppercase)
    )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="table_plane" type="plane" '
        'pos="0 0 .75" size="1 1 .1"/>' + bodies + "</worldbody></mujoco>"
    )


def row_state(model, letters):
    data = mujoco.MjData(model)
    quaternion = [2**-0.5, 0, 2**-0.5, 0]
    for letter in string.ascii_uppercase:
        adr = model.joint(f"block_{letter}_jnt").qposadr[0]
        data.qpos[adr + 3 : adr + 7] = quaternion
    for index, letter in enumerate(letters):
        adr = model.joint(f"block_{letter}_jnt").qposadr[0]
        data.qpos[adr : adr + 3] = [0.55, 0.16 - 0.04 * index, 0.765]
    return data.qpos[None].copy()


def goal_state(sequence, seed=37):
    return RandomizationState(seed, {}, metadata=SpellingGoal(sequence, seed).to_metadata())


class SpellingGoalTest(unittest.TestCase):
    def test_same_seed_repeats_across_interleaved_lengths(self):
        expected = {length: SpellingGoal.sample(length, 976) for length in (2, 3, 4)}
        for length in (4, 2, 3, 2, 4, 3):
            self.assertEqual(SpellingGoal.sample(length, 976), expected[length])

    def test_seeds_cover_feasible_unique_letter_sequences(self):
        for length in (2, 3, 4):
            goals = [SpellingGoal.sample(length, seed) for seed in range(100)]
            self.assertGreater(len({goal.sequence for goal in goals}), 80)
            self.assertEqual(set("".join(goal.sequence for goal in goals)), set(string.ascii_uppercase))
            for goal in goals:
                self.assertEqual(len(goal.sequence), length)
                self.assertEqual(len(set(goal.sequence)), length)

    def test_goal_roundtrip_preserves_content_prompt_and_digest(self):
        goal = SpellingGoal.sample(4, 977)
        metadata = json.loads(json.dumps(goal.to_metadata()))
        self.assertEqual(SpellingGoal.from_metadata(metadata), goal)
        self.assertIn(goal.sequence.lower(), metadata["prompt"])

    def test_corrupt_goal_or_prompt_is_rejected(self):
        goal = SpellingGoal("XYZ", 31)
        for field, value in (("sequence", "ABC"), ("seed", 32), ("digest", "0" * 64)):
            with self.subTest(field=field):
                metadata = deepcopy(goal.to_metadata())
                metadata["spelling_goal"][field] = value
                with self.assertRaises(ValueError):
                    SpellingGoal.from_metadata(metadata)
        metadata = goal.to_metadata()
        metadata["prompt"] = "spell abc"
        with self.assertRaises(ValueError):
            SpellingGoal.from_metadata(metadata)

    def test_invalid_goal_specs_fail(self):
        for length in (1, 5, True, 3.0, None):
            with self.subTest(length=length), self.assertRaises(ValueError):
                SpellingGoal.sample(length, 31)
        for seed in (-1, True, None, 1.0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                SpellingGoal.sample(3, seed)
        for word in ("AA", "A1", "aB", "AÉ", "ABCDE"):
            with self.subTest(word=word), self.assertRaises(ValueError):
                SpellingGoal(word, 31)
        with self.assertRaises(ValueError):
            SpellingGoal("ABC", 31, generator_revision="unknown")

    def test_all_new_task_specs_bind_matching_lengths(self):
        for length in (2, 3, 4):
            spec = get_task_spec(f"spell_random_{length}")
            self.assertEqual(spec.reset_request_options(), {"sequence_length": length})
            self.assertEqual(spec.evaluator_kwargs(), {"sequence_length": length})
            self.assertEqual(spec.env_task, "blocks")

    def test_fixed_words_keep_fixed_targets(self):
        for word in ("cat", "dog", "fish", "bair", "xdof", "abc", "yam", "agi"):
            spec = get_task_spec(f"spell_{word}")
            self.assertEqual(spec.evaluator_kwargs(), {"word": word})
            self.assertIsNone(spec.reset_request_options())


class BlocksGoalRandomizationTest(unittest.TestCase):
    def test_goal_stream_does_not_change_the_requested_pose_seed(self):
        state = RandomizationState(976, {"example": {"pos": [1, 2, 3]}})
        with patch.object(SceneRandomizer, "randomize", return_value=state) as base:
            actual = BlocksRandomizer().randomize(
                None, None, seed=976, request={"sequence_length": 3, "randomize_scales": False}
            )
        base.assert_called_once_with(None, None, seed=976, request={"randomize_scales": False})
        self.assertEqual(actual.object_states, {"example": {"pos": [1, 2, 3]}})
        self.assertEqual(SpellingGoal.from_metadata(actual.metadata), SpellingGoal.sample(3, 976))
        restored = RandomizationState.from_dict(json.loads(json.dumps(actual.to_dict())))
        self.assertEqual(SpellingGoal.from_metadata(restored.metadata), SpellingGoal.sample(3, 976))

    def test_fixed_word_reset_does_not_add_a_goal(self):
        state = RandomizationState(976, {})
        with patch.object(SceneRandomizer, "randomize", return_value=state) as base:
            actual = BlocksRandomizer().randomize(None, None, seed=976)
        base.assert_called_once_with(None, None, seed=976, request=None)
        self.assertEqual(actual.metadata, {})

    def test_invalid_request_fails_before_pose_randomization(self):
        for request in ("ABC", {"sequence_length": 3, "typo": 1}, {"sequence_length": 7}):
            with self.subTest(request=request):
                with patch.object(SceneRandomizer, "randomize") as base:
                    with self.assertRaises((TypeError, ValueError)):
                        BlocksRandomizer().randomize(None, None, seed=31, request=request)
                    base.assert_not_called()


class SpellingEvaluatorGoalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = block_model()

    def evaluator(self, length=3):
        return SpellWordEvaluator(
            model=self.model, spec=get_task_spec(f"spell_random_{length}"), sequence_length=length
        )

    def test_requires_a_goal_after_every_reset(self):
        evaluator = self.evaluator()
        row = row_state(self.model, "ABC")
        with self.assertRaisesRegex(RuntimeError, "after every reset"):
            evaluator.evaluate_qpos_batch(row)
        evaluator.configure_from_randomization(self.model, goal_state("ABC"))
        self.assertTrue(evaluator.evaluate_qpos_batch(row).success[0])
        evaluator.reset()
        with self.assertRaisesRegex(RuntimeError, "after every reset"):
            evaluator.evaluate_qpos_batch(row)

    def test_reused_model_rebinds_target_joints_for_every_goal(self):
        evaluator = self.evaluator()
        abc_row = row_state(self.model, "ABC")
        evaluator.configure_from_randomization(self.model, goal_state("ABC"))
        self.assertTrue(evaluator.evaluate_qpos_batch(abc_row).success[0])
        evaluator.reset()
        evaluator.configure_from_randomization(self.model, goal_state("XYZ"))
        self.assertFalse(evaluator.evaluate_qpos_batch(abc_row).success[0])
        self.assertTrue(evaluator.evaluate_qpos_batch(row_state(self.model, "XYZ")).success[0])
        evaluator.reset()
        evaluator.configure_from_randomization(self.model, goal_state("ABC"))
        self.assertTrue(evaluator.evaluate_qpos_batch(abc_row).success[0])

    def test_fresh_model_and_serialized_goal_reproduce_scores(self):
        state = RandomizationState.from_dict(json.loads(json.dumps(goal_state("QXZ").to_dict())))
        first = self.evaluator()
        first.configure_from_randomization(self.model, state)
        other_model = block_model()
        second = SpellWordEvaluator(model=other_model, spec=get_task_spec("spell_random_3"), sequence_length=3)
        second.configure_from_randomization(other_model, state)
        for letters in ("QXZ", "AXZ", "QAZ", "QXA", "ZXQ"):
            a = first.evaluate_qpos_batch(row_state(self.model, letters))
            b = second.evaluate_qpos_batch(row_state(other_model, letters))
            np.testing.assert_array_equal(a.reward, b.reward)
            np.testing.assert_array_equal(a.success, b.success)

    def test_goal_length_must_match_the_selected_task(self):
        with self.assertRaisesRegex(ValueError, "length differs"):
            self.evaluator(2).configure_from_randomization(self.model, goal_state("ABC"))

    def test_fixed_word_rejects_random_goal(self):
        evaluator = SpellWordEvaluator(model=self.model, spec=get_task_spec("spell_abc"), word="abc")
        with self.assertRaisesRegex(ValueError, "fixed-word"):
            evaluator.configure_from_randomization(self.model, goal_state("XYZ"))

    def test_fixed_word_reward_and_binary_success_remain_native(self):
        evaluator = SpellWordEvaluator(model=self.model, spec=get_task_spec("spell_abc"), word="abc")
        for letters, reward, success in (
            ("ABC", 1.0, True),
            ("XBC", 0.8, False),
            ("AXC", 0.6, False),
            ("ABX", 0.8, False),
        ):
            with self.subTest(letters=letters):
                result = evaluator.evaluate_qpos_batch(row_state(self.model, letters))
                self.assertAlmostEqual(float(result.reward[0]), reward, places=6)
                self.assertEqual(bool(result.success[0]), success)

    def test_configured_goal_supports_batch_resize(self):
        evaluator = self.evaluator()
        evaluator.configure_from_randomization(self.model, goal_state("ABC"))
        rows = np.concatenate((row_state(self.model, "ABC"), row_state(self.model, "XBC")))
        result = evaluator.evaluate_qpos_batch(rows)
        np.testing.assert_array_equal(result.success, [True, False])


if __name__ == "__main__":
    unittest.main()
