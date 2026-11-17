"""Seeded spelling goals through native reset, evaluation, and world metrics."""

import json
import string
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from abc_sim import make_env
from abc_sim.batched_env import PerWorldEvaluators
from abc_sim.randomization.core import RandomizationState
from abc_sim.randomization.tasks.blocks import BlocksRandomizer
from abc_sim.spelling_goal import SpellingGoal
from abc_sim.task_eval.registry import make_task_evaluator
from abc_sim.task_specs import SimTaskSpec, get_task_spec, list_task_specs


def spelling_spec(length=3):
    return SimTaskSpec(
        name="sampled_spelling",
        env_task="blocks",
        prompt="spell the sampled letter sequence",
        evaluator_name="spell_word",
        evaluator_options=(("sequence_length", length),),
        reset_request=(("sequence_length", length),),
    )


def blocks_xml():
    blocks = "".join(
        f'<body name="block_{letter}" pos="{0.5 + 0.08 * (index // 7)} {-0.3 + 0.1 * (index % 7)} .765">'
        f'<freejoint name="block_{letter}_jnt"/>'
        '<geom type="box" size=".015 .015 .015" mass=".03"/></body>'
        for index, letter in enumerate(string.ascii_uppercase)
    )
    # The environment maps these joints into its regular two-arm observation.
    joints = [
        (f"{arm}_joint{index}", f"{arm}_joint{index}")
        for arm in ("left", "right")
        for index in range(1, 7)
    ] + [(f"{arm}_left_finger", f"{arm}_gripper") for arm in ("left", "right")]
    arms = "".join(
        f'<body pos="-2 {index} 1"><joint name="{joint}"/>'
        '<geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/></body>'
        for index, (joint, _) in enumerate(joints)
    )
    actuators = "".join(
        f'<position name="{actuator}" joint="{joint}"/>' for joint, actuator in joints
    )
    return (
        '<mujoco><worldbody><geom name="table_plane" type="plane" '
        'pos="0 0 .75" size="1 1 .1"/>'
        + blocks
        + arms
        + "</worldbody><actuator>"
        + actuators
        + "</actuator></mujoco>"
    )


def row_state(model, letters):
    data = mujoco.MjData(model)
    for letter in string.ascii_uppercase:
        adr = model.joint(f"block_{letter}_jnt").qposadr[0]
        data.qpos[adr + 3 : adr + 7] = [2**-0.5, 0, 2**-0.5, 0]
    for index, letter in enumerate(letters):
        adr = model.joint(f"block_{letter}_jnt").qposadr[0]
        data.qpos[adr : adr + 3] = [0.55, 0.16 - 0.04 * index, 0.765]
    return data.qpos[None].copy()


def goal_state(sequence, seed=37):
    return RandomizationState(
        seed, {}, metadata=SpellingGoal(sequence, seed).to_metadata()
    )


class SpellingGoalTest(unittest.TestCase):
    def test_seed_repeats_and_samples_distinct_letters(self):
        for length in (2, 3, 4):
            goals = [SpellingGoal.sample(length, seed) for seed in range(30)]
            self.assertEqual(
                goals, [SpellingGoal.sample(length, seed) for seed in range(30)]
            )
            self.assertGreater(len({goal.sequence for goal in goals}), 25)
            for goal in goals:
                self.assertEqual(len(set(goal.sequence)), length)
                self.assertTrue(set(goal.sequence) <= set(string.ascii_uppercase))

    def test_plain_metadata_roundtrip_and_prompt_agreement(self):
        goal = SpellingGoal.sample(4, 976)
        metadata = json.loads(json.dumps(goal.to_metadata()))
        self.assertEqual(
            metadata["spelling_goal"], {"sequence": goal.sequence, "seed": 976}
        )
        self.assertEqual(SpellingGoal.from_metadata(metadata), goal)
        metadata["prompt"] = "spell a different sequence"
        with self.assertRaisesRegex(ValueError, "prompt must agree"):
            SpellingGoal.from_metadata(metadata)

    def test_invalid_goals_fail(self):
        for length in (1, 5, True, 3.0):
            with self.subTest(length=length), self.assertRaises(ValueError):
                SpellingGoal.sample(length, 31)
        for seed in (-1, True, None, 1.0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                SpellingGoal.sample(3, seed)
        for sequence in ("AA", "A1", "aB", "AÉ", "ABCDE"):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                SpellingGoal(sequence, 31)


class SpellingResetTest(unittest.TestCase):
    def test_switching_tasks_rebinds_reset_goal_prompt_and_evaluator(self):
        env = make_env(
            task=spelling_spec(2),
            scene_xml_string=blocks_xml(),
            render_cameras=False,
        )
        self.addCleanup(env.close)
        for spec in (spelling_spec(2), spelling_spec(3), get_task_spec("spell_cat")):
            with self.subTest(spec=spec):
                env.set_task(spec)
                observation, info = env.reset(seed=31)
                metadata = info["randomization"].metadata
                if spec.reset_request:
                    goal = SpellingGoal.from_metadata(metadata)
                    self.assertEqual(
                        len(goal.sequence), spec.evaluator_kwargs()["sequence_length"]
                    )
                    self.assertEqual(observation["prompt"], goal.prompt)
                    word = goal.sequence
                    expected_word_metric = [word]
                else:
                    self.assertNotIn("spelling_goal", metadata)
                    self.assertEqual(observation["prompt"], "spell cat")
                    word = "CAT"
                    expected_word_metric = word
                env.data.qpos[:] = row_state(env.model, word)[0]
                result = env.evaluate_task()
                self.assertTrue(result.scalar_success())
                self.assertEqual(result.metrics["word"], expected_word_metric)

    def test_explicit_prompt_override_returns_after_randomized_task(self):
        for prompt in ("use the caller's spelling instruction", ""):
            with self.subTest(prompt=prompt):
                env = make_env(
                    task=spelling_spec(2),
                    prompt=prompt,
                    scene_xml_string=blocks_xml(),
                    render_cameras=False,
                )
                self.addCleanup(env.close)
                self.assertEqual(env.prompt, prompt)
                observation, info = env.reset(seed=31)
                goal = SpellingGoal.from_metadata(info["randomization"].metadata)
                self.assertEqual(observation["prompt"], goal.prompt)
                env.set_task("spell_cat")
                self.assertEqual(env.prompt, prompt)
                observation, info = env.reset(seed=31)
                self.assertEqual(observation["prompt"], prompt)
                self.assertNotIn("spelling_goal", info["randomization"].metadata)

    def test_goal_sampling_preserves_default_pose_randomization(self):
        model = mujoco.MjModel.from_xml_string(blocks_xml())
        data = mujoco.MjData(model)
        randomizer = BlocksRandomizer()
        for seed in (31, 976):
            mujoco.mj_resetData(model, data)
            default = randomizer.randomize(model, data, seed=seed)
            expected_qpos = data.qpos.copy()
            self.assertNotIn("spelling_goal", default.metadata)
            for length in (2, 3, 4):
                mujoco.mj_resetData(model, data)
                state = randomizer.randomize(
                    model, data, seed=seed, request={"sequence_length": length}
                )
                np.testing.assert_array_equal(data.qpos, expected_qpos)
                self.assertEqual(state.object_states, default.object_states)
                self.assertEqual(
                    SpellingGoal.from_metadata(state.metadata),
                    SpellingGoal.sample(length, seed),
                )

    def test_custom_spec_make_env_reset_prompt_and_evaluator_agree(self):
        catalogue = list_task_specs()
        for length in (2, 3, 4):
            env = make_env(
                task=spelling_spec(length),
                scene_xml_string=blocks_xml(),
                render_cameras=False,
            )
            self.addCleanup(env.close)
            for seed in (31, 976, 31):
                obs, info = env.reset(seed=seed)
                state = info["randomization"]
                restored = RandomizationState.from_dict(
                    json.loads(json.dumps(state.to_dict()))
                )
                goal = SpellingGoal.from_metadata(restored.metadata)
                self.assertEqual(goal, SpellingGoal.sample(length, seed))
                self.assertEqual(obs["prompt"], goal.prompt)
                env.data.qpos[:] = row_state(env.model, goal.sequence)[0]
                result = env.evaluate_task()
                self.assertEqual(result.metrics["word"], [goal.sequence])
                self.assertTrue(result.scalar_success())
                self.assertEqual(result.scalar_reward(), 1.0)
            self.assertIsNone(env.max_episode_steps)
        self.assertEqual(list_task_specs(), catalogue)
        self.assertFalse(any(spec.name == "sampled_spelling" for spec in catalogue))
        for word in ("cat", "dog", "fish", "bair", "xdof", "abc", "yam", "agi"):
            spec = get_task_spec(f"spell_{word}")
            self.assertEqual(spec.evaluator_kwargs(), {"word": word})
            self.assertIsNone(spec.reset_request_options())


class SpellingEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(blocks_xml())

    def test_rebinds_reused_and_new_models_and_requires_goal_after_reset(self):
        evaluator = make_task_evaluator(self.model, spelling_spec())
        with self.assertRaisesRegex(RuntimeError, "after every reset"):
            evaluator.evaluate_qpos_batch(row_state(self.model, "ABC"))
        for model, word in (
            (self.model, "ABC"),
            (self.model, "XYZ"),
            (mujoco.MjModel.from_xml_string(blocks_xml()), "ABC"),
        ):
            evaluator.configure_from_randomization(model, goal_state(word))
            self.assertTrue(
                evaluator.evaluate_qpos_batch(row_state(model, word)).scalar_success()
            )
            other = "XYZ" if word == "ABC" else "ABC"
            self.assertFalse(
                evaluator.evaluate_qpos_batch(row_state(model, other)).scalar_success()
            )
            evaluator.reset()
            with self.assertRaisesRegex(RuntimeError, "after every reset"):
                evaluator.evaluate_qpos_batch(row_state(model, word))

    def test_invalid_configuration_and_goal_length_fail(self):
        evaluator = make_task_evaluator(self.model, spelling_spec(2))
        with self.assertRaisesRegex(ValueError, "length differs"):
            evaluator.configure_from_randomization(self.model, goal_state("ABC"))
        fixed = make_task_evaluator(self.model, get_task_spec("spell_abc"))
        with self.assertRaisesRegex(ValueError, "fixed-word"):
            fixed.configure_from_randomization(self.model, goal_state("XYZ"))

    def test_fixed_word_scalar_metric_and_native_reward_are_unchanged(self):
        evaluator = make_task_evaluator(self.model, get_task_spec("spell_abc"))
        for letters, reward, success in (
            ("ABC", 1.0, True),
            ("XBC", 0.8, False),
            ("AXC", 0.6, False),
            ("ABX", 0.8, False),
        ):
            result = evaluator.evaluate_qpos_batch(row_state(self.model, letters))
            self.assertEqual(result.metrics["word"], "ABC")
            self.assertAlmostEqual(result.scalar_reward(), reward, places=6)
            self.assertEqual(result.scalar_success(), success)

    def test_shared_goal_supports_multiple_worlds(self):
        evaluator = make_task_evaluator(self.model, spelling_spec())
        evaluator.configure_from_randomization(self.model, goal_state("ABC"))
        rows = np.concatenate([row_state(self.model, word) for word in ("ABC", "XBC")])
        result = evaluator.evaluate_qpos_batch(rows)
        np.testing.assert_array_equal(result.success, [True, False])
        self.assertEqual(result.metrics["word"], ["ABC", "ABC"])


class PerWorldSpellingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(blocks_xml())

    def rows(self, words):
        return np.concatenate([row_state(self.model, word) for word in words])

    def test_different_goals_survive_metric_discovery_and_json_world_selection(self):
        for words in (("AB", "XY"), ("ABC", "XYZ"), ("ABCD", "WXYZ")):
            wrapper = PerWorldEvaluators(self.model, spelling_spec(len(words[0])), 2)
            wrapper.reset(self.model, [goal_state(word) for word in words])
            rows = self.rows(words)
            keys = wrapper.per_world_keys(rows)
            self.assertIn("word", keys)
            result = wrapper.evaluate_qpos_batch(rows)
            np.testing.assert_array_equal(result.success, [True, True])
            self.assertEqual(result.metrics["word"], list(words))
            info = json.loads(json.dumps(result.to_info()))
            for index, word in enumerate(words):
                # Application consumers select the declared world axis before serialization.
                record = {
                    key: value[index] if key in keys else value
                    for key, value in info.items()
                }
                self.assertEqual(record["word"], word)

    def test_probe_does_not_reset_goal_or_change_live_success_history(self):
        wrapper = PerWorldEvaluators(self.model, spelling_spec(), 1)
        wrapper.reset(self.model, [goal_state("ABC")])
        live = wrapper.evaluators[0]
        with (
            patch.object(live, "reset", side_effect=AssertionError("live reset")),
            patch.object(
                live, "evaluate_qpos_batch", side_effect=AssertionError("live score")
            ),
        ):
            wrapper.per_world_keys(self.rows(["ABC"]))
        failed = wrapper.evaluate_qpos_batch(self.rows(["XBC"]))
        np.testing.assert_array_equal(failed.metrics["ever_success"], [False])
        wrapper.evaluate_qpos_batch(self.rows(["ABC"]))
        wrapper.per_world_keys(self.rows(["XBC"]))
        failed = wrapper.evaluate_qpos_batch(self.rows(["XBC"]))
        np.testing.assert_array_equal(failed.metrics["ever_success"], [True])

    def test_second_reset_rebinds_goals_and_clears_each_world_history(self):
        wrapper = PerWorldEvaluators(self.model, spelling_spec(), 2)
        rows = self.rows(["ABC", "XYZ"])
        wrapper.reset(self.model, [goal_state(word) for word in ("ABC", "XYZ")])
        np.testing.assert_array_equal(
            wrapper.evaluate_qpos_batch(rows).success, [True, True]
        )
        wrapper.reset(self.model, [goal_state(word) for word in ("XYZ", "ABC")])
        wrapper.per_world_keys(rows)
        result = wrapper.evaluate_qpos_batch(rows)
        self.assertEqual(result.metrics["word"], ["XYZ", "ABC"])
        np.testing.assert_array_equal(result.success, [False, False])
        np.testing.assert_array_equal(result.metrics["ever_success"], [False, False])

    def test_fixed_word_remains_one_scalar_for_all_worlds(self):
        wrapper = PerWorldEvaluators(self.model, get_task_spec("spell_abc"), 2)
        wrapper.reset(self.model, [None, None])
        rows = self.rows(["ABC", "XBC"])
        self.assertNotIn("word", wrapper.per_world_keys(rows))
        result = wrapper.evaluate_qpos_batch(rows)
        self.assertEqual(result.metrics["word"], "ABC")
        np.testing.assert_allclose(result.reward, [1.0, 0.8])


if __name__ == "__main__":
    unittest.main()
