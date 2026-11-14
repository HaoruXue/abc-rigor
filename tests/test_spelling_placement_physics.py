"""CPU contact tests runnable with unittest in the declared ABC simulator environment."""

import importlib.util
import string
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from abc_sim.spelling_placement import FreePlacementSpellingCfg, SpellingTableSupport


@unittest.skipUnless(
    importlib.util.find_spec("mujoco"),
    "MuJoCo is required for the contact-boundary CPU tests.",
)
class SpellingPhysicsTests(unittest.TestCase):
    def setUp(self):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string("""
            <mujoco>
              <option timestep="0.002"/>
              <worldbody>
                <geom name="table_plane" type="plane" size="1 1 0.1"/>
                <body name="block_A" pos="0 0 0.012">
                  <freejoint/>
                  <geom type="box" size="0.01 0.01 0.01" mass="0.05"/>
                </body>
                <body name="left_pad" mocap="true" pos="-0.1 0 0.012">
                  <geom name="left_finger" type="box" size="0.005 0.02 0.008"/>
                </body>
                <body name="right_pad" mocap="true" pos="0.1 0 0.012">
                  <geom name="right_finger" type="box" size="0.005 0.02 0.008"/>
                </body>
                <body name="unused_finger" pos="10 0 0">
                  <geom name="unused_finger_geom" type="box" size=".01 .01 .01"/>
                </body>
              </worldbody>
            </mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        self.tracker = SpellingTableSupport(
            FreePlacementSpellingCfg(),
            self.model,
            body_names=["block_A"],
            robot_geom_ids={
                self.model.geom(name).id for name in ("left_finger", "right_finger")
            },
        )

    def advance(self, steps):
        result = None
        for _ in range(steps):
            self.mujoco.mj_step(self.model, self.data)
            result = self.tracker.sample(self.data)
        return result

    def test_support_requires_settling_and_is_lost_on_lift(self):
        self.assertEqual(self.advance(20)["eligible"], [False])
        measured = self.advance(500)
        self.assertEqual(measured["eligible"], [True])
        self.assertAlmostEqual(measured["weight_fraction"][0], 1, delta=0.01)
        self.data.qpos[2] += 0.05
        self.assertEqual(self.advance(1)["eligible"], [False])

    def test_table_supported_gripped_block_is_excluded_until_release(self):
        self.assertEqual(self.advance(500)["eligible"], [True])
        self.data.mocap_pos[0, 0], self.data.mocap_pos[1, 0] = -0.0145, 0.0145
        held = self.advance(500)
        self.assertGreater(held["weight_fraction"][0], 0.5)
        self.assertGreater(
            held["robot_normal_load"][0], self.tracker.cfg.robot_force_tolerance
        )
        self.assertEqual(held["eligible"], [False])
        self.data.mocap_pos[0, 0], self.data.mocap_pos[1, 0] = -0.1, 0.1
        self.assertEqual(self.advance(1)["eligible"], [False])
        released = self.advance(300)
        self.assertEqual(released["eligible"], [True])
        self.assertEqual(released["robot_normal_load"], [0.0])

    def test_duplicate_and_skipped_physics_samples_fail(self):
        self.advance(1)
        with self.assertRaisesRegex(ValueError, "every native physics step"):
            self.tracker.sample(self.data)
        self.mujoco.mj_step(self.model, self.data)
        self.mujoco.mj_step(self.model, self.data)
        with self.assertRaisesRegex(ValueError, "every native physics step"):
            self.tracker.sample(self.data)

    def test_stationary_block_hovering_19mm_has_no_table_support(self):
        self.data.qpos[2] = 0.01 + 0.019
        self.mujoco.mj_forward(self.model, self.data)
        measured = self.tracker.sample(self.data)
        self.assertEqual(measured["linear_speed"], [0.0])
        self.assertEqual(measured["weight_fraction"], [0.0])
        self.assertEqual(measured["eligible"], [False])

    def test_load_from_an_untracked_contact_is_not_excluded_by_v1(self):
        self.advance(500)
        self.data.mocap_pos[0, 0], self.data.mocap_pos[1, 0] = -0.0145, 0.0145
        tracked = self.advance(500)
        self.assertGreater(
            tracked["robot_normal_load"][0], self.tracker.cfg.robot_force_tolerance
        )
        self.assertEqual(tracked["eligible"], [False])
        self.tracker = SpellingTableSupport(
            FreePlacementSpellingCfg(),
            self.model,
            body_names=["block_A"],
            robot_geom_ids={self.model.geom("unused_finger_geom").id},
        )
        untracked = self.advance(500)
        self.assertGreater(untracked["weight_fraction"][0], 0.5)
        self.assertEqual(untracked["robot_normal_load"], [0.0])
        self.assertEqual(untracked["eligible"], [True])


@unittest.skipUnless(
    importlib.util.find_spec("mujoco"),
    "MuJoCo is required for native evaluator CPU tests.",
)
class NativeSpellingPlacementTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        from abc_sim.env import MuJoCoYAMEnv
        from abc_sim.randomization.core import RandomizationState
        from abc_sim.spelling_goal import SpellingGoal
        from abc_sim.task_eval.registry import make_task_evaluator
        from abc_sim.task_variants import free_placement_spelling_task

        bodies = []
        for index, letter in enumerate(string.ascii_uppercase):
            bodies.append(
                f'<body name="block_{letter}" pos="{0.4 + 0.1 * (index // 9)} {0.4 - 0.085 * (index % 9)} .012">'
                f'<freejoint name="block_{letter}_jnt"/>'
                '<geom type="box" size=".01 .01 .01" mass=".05"/></body>'
            )
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".002"/><worldbody>'
            '<geom name="table_plane" type="plane" size="1 1 .1"/>'
            '<body name="left_link_left_finger" pos="10 0 0">'
            '<geom name="finger_geom" type="box" size=".01 .01 .01"/></body>'
            '<body name="left_link_6" pos="11 0 0">'
            '<geom name="wrist_geom" type="box" size=".01 .01 .01"/></body>'
            '<body name="wrist_camera" pos="12 0 0">'
            '<geom name="camera_geom" type="box" size=".01 .01 .01"/></body>'
            + "".join(bodies)
            + "</worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        self.evaluator = make_task_evaluator(self.model, free_placement_spelling_task())
        self.evaluator.configure_from_randomization(
            self.model,
            RandomizationState(37, {}, metadata=SpellingGoal("AB", 37).to_metadata()),
        )
        self.evaluator.after_reset(self.model, self.data)
        self.env = object.__new__(MuJoCoYAMEnv)
        self.env.model, self.env.data = self.model, self.data
        self.env._task_runtime = None
        self.env._task_evaluator = self.evaluator
        self.env._physics_observer = self.evaluator.after_physics_step
        self.env._task_spec = self.evaluator.spec
        self.env._control_decimation = 34
        self.env._task_start_time = float(self.data.time)
        self.env.cur_step = 0
        self.env.max_episode_steps = None
        self.env.terminate_on_success = False

    def test_physics_hook_samples_each_tick_scores_every_seventeen_and_reads_are_pure(
        self,
    ):
        with (
            patch.object(
                self.evaluator._tracker, "sample", wraps=self.evaluator._tracker.sample
            ) as support,
            patch.object(
                self.evaluator,
                "_score_model_data",
                wraps=self.evaluator._score_model_data,
            ) as scoring,
        ):
            self.env.advance_physics(16)
            with self.assertRaisesRegex(RuntimeError, "first scheduled"):
                self.env.evaluate_task()
            self.env.advance_physics(1)
            first = self.env.evaluate_task()
            self.assertEqual(first.metrics["verifier_sample_count"], 1)
            self.assertEqual(support.call_count, 17)
            self.assertEqual(scoring.call_count, 1)
            for _ in range(4):
                self.assertIs(self.env.evaluate_task(), first)
            self.assertEqual(support.call_count, 17)
            self.assertEqual(scoring.call_count, 1)
            self.env.advance_physics()
            self.assertEqual(support.call_count, 51)
            self.assertEqual(scoring.call_count, 3)
            self.assertEqual(
                self.env.evaluate_task().metrics["verifier_sample_count"], 3
            )

    def test_native_score_and_conservative_placement_are_separate(self):
        for index, letter in enumerate("AB"):
            address = self.model.joint(f"block_{letter}_jnt").qposadr[0]
            self.data.qpos[address : address + 3] = [0, 0.1 - 0.04 * index, 0.012]
            self.data.qpos[address + 3 : address + 7] = [2**-0.5, 0, 2**-0.5, 0]
        self.env.advance_physics(17)
        early = self.env.evaluate_task()
        self.assertEqual(float(early.reward[0]), 0)
        self.assertEqual(float(early.metrics["original_spell_reward"][0]), 1)
        self.env.advance_physics(493)
        complete = self.env.evaluate_task()
        self.assertEqual(float(complete.reward[0]), 1)
        self.assertTrue(complete.success[0])
        evidence = complete.metrics["spelling_placement"]
        self.assertEqual(
            evidence["frame"]["supported"], evidence["support"]["eligible"]
        )
        self.assertEqual(evidence["score"]["reward"], 1)
        self.assertEqual(evidence["contact_scope"], "finger-subtrees-v1")

    def test_reset_clears_support_and_requires_rebinding_even_on_reused_model(self):
        from abc_sim.randomization.core import RandomizationState
        from abc_sim.spelling_goal import SpellingGoal

        self.env.advance_physics(17)
        self.evaluator.reset()
        with self.assertRaisesRegex(RuntimeError, "realized spelling goal"):
            self.evaluator.after_reset(self.model, self.data)
        self.evaluator.configure_from_randomization(
            self.model,
            RandomizationState(38, {}, metadata=SpellingGoal("XY", 38).to_metadata()),
        )
        self.evaluator.after_reset(self.model, self.data)
        with self.assertRaisesRegex(RuntimeError, "first scheduled"):
            self.env.evaluate_task()
        self.env.advance_physics(17)
        self.assertEqual(self.env.evaluate_task().metrics["word"], "XY")
        self.assertEqual(self.env.evaluate_task().metrics["verifier_sample_count"], 1)

    def test_native_reset_binds_after_settling_without_sampling_the_previous_episode(
        self,
    ):
        from abc_sim.randomization.core import RandomizationState
        from abc_sim.spelling_goal import SpellingGoal

        self.env.advance_physics(17)
        previous_tracker = self.evaluator._tracker
        requests = []

        def randomize(model, data, *, seed, request):
            self.assertIsNone(self.env._physics_observer)
            requests.append(request)
            self.env.advance_physics(5)
            return RandomizationState(
                seed, {}, metadata=SpellingGoal("XY", seed).to_metadata()
            )

        self.env._task_randomizer = SimpleNamespace(randomize=randomize)
        with (
            patch.object(self.env, "_get_reset_arm_state", return_value=None),
            patch.object(self.env, "_set_qpos_from_state"),
            patch.object(self.env, "_reset_camera_backend"),
            patch.object(self.env, "get_obs", return_value={}),
            patch.object(
                previous_tracker,
                "sample",
                side_effect=AssertionError("old observer ran"),
            ),
        ):
            self.env.reset(seed=38)
        self.assertEqual(requests, [{"sequence_length": 2}])
        self.assertIsNot(self.evaluator._tracker, previous_tracker)
        self.assertIsNone(self.evaluator._tracker.previous_time)
        self.assertEqual(self.env._task_start_time, float(self.data.time))
        self.env.advance_physics(1)
        self.assertEqual(self.evaluator._tracker.sustained.tolist(), [0] * 26)
        with self.assertRaisesRegex(RuntimeError, "first scheduled"):
            self.env.evaluate_task()

    def test_qpos_and_warp_capability_boundaries_fail_explicitly(self):
        from abc_sim.batched_env import BatchedWarpYAMEnv

        with self.assertRaisesRegex(NotImplementedError, "solved contacts"):
            self.evaluator.evaluate_qpos_batch(self.data.qpos[None])
        with self.assertRaisesRegex(NotImplementedError, "solved contact"):
            BatchedWarpYAMEnv(self.env, num_worlds=1, camera_height=8, camera_width=8)

    def test_historical_contact_scope_excludes_fingers_but_not_wrist_or_camera(self):
        tracked = self.evaluator._tracker.robot_geom_ids
        self.assertIn(self.model.geom("finger_geom").id, tracked)
        self.assertNotIn(self.model.geom("wrist_geom").id, tracked)
        self.assertNotIn(self.model.geom("camera_geom").id, tracked)

    def test_ordinary_evaluator_keeps_the_unobserved_native_loop(self):
        self.env._task_evaluator = SimpleNamespace()
        self.env._physics_observer = None
        with patch.object(
            self.env, "_advance_native_physics", wraps=self.env._advance_native_physics
        ) as native:
            self.env.advance_physics()
        native.assert_called_once_with(34)

    def test_horizon_uses_elapsed_physics_time_and_preserves_original_defaults(self):
        for elapsed, expected in ((59.999, False), (60, True)):
            self.data.time = self.env._task_start_time + elapsed
            info = {}
            self.assertEqual(self.env._episode_status(info), (False, expected))
            self.assertEqual(info["task_horizon_seconds"], 60)
        from abc_sim.task_specs import get_task_spec

        self.env._task_spec = get_task_spec("spell_random_2")
        self.assertEqual(self.env._episode_status({}), (False, False))


if __name__ == "__main__":
    unittest.main()
