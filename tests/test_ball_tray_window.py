"""Ball tray, default scoring: hold 10-15 s, then a gentle placement with the ball still on the tray."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from abc_sim.task_eval.ball_tray import BallTrayBalancingEvaluator
from abc_sim.task_runtime.ball_tray import BallTrayBalancingRuntime
from abc_sim.task_specs import get_task_spec

REST_Z, UP_Z, LOW_Z = (
    0.009,
    0.109,
    0.025,
)  # on the table, 10 cm up, 1.6 cm up (below the 5 cm lift)
HANDLE_Y = 0.16
AWAY = [1.0, 1.0, 0.5]

XML = f"""
<mujoco><worldbody>
  <geom name="table_plane" type="plane" size="1 1 .1"/>
  <body name="ball_tray" pos="0 0 {REST_Z}"><freejoint name="ball_tray_joint"/>
    <geom type="box" size=".15 .15 .01" mass="1"/>
    <geom name="ball_tray_left_handle_collision" type="box" size=".03 .01 .01" pos="0 {HANDLE_Y} .02"/>
    <geom name="ball_tray_right_handle_collision" type="box" size=".03 .01 .01" pos="0 -{HANDLE_Y} .02"/>
    <site name="ball_tray_center_site"/></body>
  <body name="ball_tray_ball" pos="0 0 {REST_Z + 0.03}"><freejoint name="ball_tray_ball_joint"/>
    <geom type="sphere" size=".01" mass=".1"/></body>
  <body name="left_link_left_finger" pos="{AWAY[0]} {AWAY[1]} {AWAY[2]}"><freejoint name="left_finger_joint"/>
    <geom type="box" size=".01 .01 .01" mass=".1"/></body>
  <body name="right_link_left_finger" pos="-1 -1 .5"><freejoint name="right_finger_joint"/>
    <geom type="box" size=".01 .01 .01" mass=".1"/></body>
</worldbody></mujoco>
"""

PROMPT = "keep the ball balanced on the tray for 10 to 15 seconds, then gently place the tray on the table"


class BallTrayWindowTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.use_evaluator()

    def use_evaluator(self, **options):
        self.evaluator = BallTrayBalancingEvaluator(
            model=self.model, spec=get_task_spec("ball_tray_balancing"), **options
        )
        self.evaluator.configure_from_randomization(
            self.model, SimpleNamespace(seed=0, metadata={"hold_duration_s": 7.0})
        )
        self.sample(0, REST_Z)

    def put(self, joint, xyz):
        address = int(self.model.jnt_qposadr[self.model.joint(joint).id])
        self.data.qpos[address : address + 3] = xyz

    def sample(self, t, z, *, held=True, ball_inside=True, touch_ball=False):
        self.put("ball_tray_joint", [0, 0, z])
        ball = [0 if ball_inside else 1, 0, z + 0.03]
        self.put("ball_tray_ball_joint", ball)
        self.put("left_finger_joint", [0, HANDLE_Y, z + 0.02] if held else AWAY)
        self.put(
            "right_finger_joint", [0, -HANDLE_Y, z + 0.02] if held else [-1, -1, 0.5]
        )
        if touch_ball:
            self.put("left_finger_joint", [0, 0, z + 0.045])
        self.data.time = t
        mujoco.mj_forward(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def hold(self, start, end, step=0.5):
        result = None
        for t in np.arange(start, end + 1e-9, step):
            result = self.sample(float(t), UP_Z)
        return result

    def place(self, t, **kwargs):
        """Lower from 10 cm in 0.6 s (about 0.17 m/s) and touch down at ``t``."""
        self.sample(t - 0.1, LOW_Z, **kwargs)
        return self.sample(t, REST_Z, **kwargs)

    @staticmethod
    def score(result):
        return result.scalar_success(), round(float(result.reward[0]), 4)

    def make_runtime(self):
        runtime = BallTrayBalancingRuntime()
        runtime.bind(self.model, self.data)
        runtime.after_reset(
            self.model, self.data, randomization=SimpleNamespace(seed=0)
        )
        runtime.on_task_evaluation(self.sample(0, REST_Z))
        runtime.after_step(self.model, self.data)
        return runtime

    def runtime_sample(self, runtime, t, z, **kwargs):
        runtime.before_step(self.model, self.data)
        result = self.sample(t, z, **kwargs)
        runtime.on_task_evaluation(result)
        runtime.after_step(self.model, self.data)
        return result

    def test_hold_in_the_window_then_gentle_placement_succeeds(self):
        self.assertEqual(
            self.score(self.hold(1, 6)), (False, 0.45)
        )  # half of the 10 s minimum
        held = self.hold(6.5, 12)
        self.assertTrue(held.metrics["hold_complete"][0])
        self.assertEqual(self.score(held), (False, 0.9))
        touchdown = self.place(12.6)
        self.assertTrue(touchdown.metrics["settling"][0])
        self.assertEqual(self.score(touchdown), (False, 0.9))
        self.assertFalse(
            self.sample(13.5, REST_Z).scalar_success()
        )  # the ball must stay 1 s
        done = self.sample(13.6, REST_Z, held=False)
        self.assertEqual(self.score(done), (True, 1.0))

    def test_runtime_stops_active_and_overdue_pulses_through_settling_and_success(self):
        runtime = self.make_runtime()
        self.runtime_sample(runtime, 1.0, UP_Z)
        self.runtime_sample(runtime, 10.99, UP_Z)
        pulses = runtime.debug_state()["perturbation_count"]
        self.assertGreater(pulses, 0)
        ball_body = self.model.body("ball_tray_ball").id
        other_force = np.array([0.1, 0.2, 0.3])
        self.data.xfrc_applied[ball_body, :3] += other_force
        runtime.before_step(self.model, self.data)
        self.assertGreater(
            np.linalg.norm(self.data.xfrc_applied[ball_body, :3] - other_force), 0.0
        )

        held = self.runtime_sample(runtime, 11.0, UP_Z)
        self.assertTrue(held.metrics["hold_complete"][0])
        self.assertFalse(held.metrics["perturbations_allowed"][0])
        runtime.before_step(self.model, self.data)
        np.testing.assert_allclose(self.data.xfrc_applied[ball_body, :3], other_force)
        self.runtime_sample(runtime, 12.5, LOW_Z)
        touchdown = self.runtime_sample(runtime, 12.6, REST_Z)
        self.assertTrue(touchdown.metrics["settling"][0])

        # Runtime callbacks run at 500 Hz; the evaluator publishes a score every 17 ticks.
        for control_step in range(45):
            for tick in range(17):
                runtime.before_step(self.model, self.data)
                np.testing.assert_allclose(
                    self.data.xfrc_applied[ball_body, :3], other_force
                )
                self.data.time = 12.6 + (control_step * 17 + tick + 1) * 0.002
                if tick == 16:
                    result = self.sample(float(self.data.time), REST_Z)
                    runtime.on_task_evaluation(result)
                    self.assertTrue(result.metrics["hold_complete"][0])
                    self.assertFalse(result.metrics["perturbations_allowed"][0])
                runtime.after_step(self.model, self.data)
                self.assertEqual(runtime.debug_state()["perturbation_count"], pulses)
        self.assertEqual(self.score(result), (True, 1.0))
        self.assertTrue(runtime.debug_state()["complete"])

        self.use_evaluator()
        runtime.after_reset(
            self.model, self.data, randomization=SimpleNamespace(seed=0)
        )
        reset = self.runtime_sample(runtime, 0.0, REST_Z)
        self.assertFalse(reset.metrics["hold_complete"][0])
        self.assertTrue(reset.metrics["perturbations_allowed"][0])
        self.runtime_sample(runtime, 1.0, UP_Z)
        self.assertGreater(runtime.debug_state()["perturbation_count"], 0)

    def test_early_placements_stop_perturbations_without_completing_the_hold(self):
        for gentle in (False, True):
            with self.subTest(gentle=gentle):
                self.use_evaluator()
                runtime = self.make_runtime()
                self.runtime_sample(runtime, 1.0, UP_Z)
                self.runtime_sample(runtime, 4.0, UP_Z)
                self.runtime_sample(runtime, 4.5, LOW_Z, held=gentle)
                placed = self.runtime_sample(runtime, 4.6, REST_Z, held=gentle)
                pulses = runtime.debug_state()["perturbation_count"]
                self.assertFalse(placed.metrics["hold_complete"][0])
                self.assertFalse(placed.metrics["perturbations_allowed"][0])
                self.assertEqual(bool(placed.metrics["settling"][0]), gentle)
                for t in (4.634, 5.0, 5.6, 6.0):
                    result = self.runtime_sample(runtime, t, REST_Z)
                    self.assertFalse(result.metrics["perturbations_allowed"][0])
                    self.assertEqual(
                        runtime.debug_state()["perturbation_count"], pulses
                    )
                self.assertFalse(result.scalar_success())

    def test_contact_loss_and_relift_restart_the_full_settling_interval(self):
        for gap_z in (LOW_Z, UP_Z):
            for gap_time in (13.0, 13.6):
                with self.subTest(gap_z=gap_z, gap_time=gap_time):
                    self.use_evaluator()
                    self.hold(1, 12)
                    self.place(12.6)
                    for t in (gap_time, 14.0):
                        interrupted = self.sample(t, gap_z)
                        self.assertFalse(interrupted.metrics["tray_table_contact"][0])
                        self.assertFalse(interrupted.scalar_success())
                        self.assertTrue(interrupted.metrics["hold_complete"][0])
                        self.assertFalse(
                            interrupted.metrics["perturbations_allowed"][0]
                        )
                    recontact = self.sample(14.1, REST_Z, held=False)
                    self.assertTrue(recontact.metrics["settling"][0])
                    self.assertFalse(recontact.scalar_success())
                    self.assertFalse(
                        self.sample(15.099, REST_Z, held=False).scalar_success()
                    )
                    self.assertEqual(
                        self.score(self.sample(15.1, REST_Z, held=False)), (True, 1.0)
                    )

    def test_contact_at_the_lift_threshold_does_not_count_as_settled(self):
        self.hold(1, 12)
        self.place(12.6)
        address = int(self.model.jnt_qposadr[self.model.joint("ball_tray_joint").id])
        self.data.qpos[address + 3 : address + 7] = [
            np.cos(np.pi / 8),
            0,
            np.sin(np.pi / 8),
            0,
        ]
        relifted = self.sample(13.6, REST_Z + 0.05)
        self.assertTrue(relifted.metrics["tray_table_contact"][0])
        self.assertTrue(relifted.metrics["lifted"][0])
        self.assertFalse(relifted.scalar_success())
        self.data.qpos[address + 3 : address + 7] = [1, 0, 0, 0]
        self.assertFalse(self.sample(13.7, REST_Z).scalar_success())
        self.assertTrue(self.sample(14.7, REST_Z).scalar_success())

    def test_ball_loss_during_a_settling_interruption_still_fails(self):
        self.hold(1, 12)
        self.place(12.6)
        self.sample(13.0, LOW_Z)
        lost = self.sample(13.1, LOW_Z, ball_inside=False)
        self.assertTrue(lost.metrics["ball_lost_after_hold"][0])
        self.assertEqual(self.score(lost), (False, 0.0))
        self.assertEqual(self.score(self.sample(20.0, REST_Z)), (False, 0.0))

    def test_settling_interruption_is_independent_per_world(self):
        self.use_evaluator(require_handles_at_touchdown=False)
        rest = self.data.qpos.copy()
        self.sample(1.0, UP_Z)
        up = self.data.qpos.copy()
        self.evaluator.reset(nworld=2)
        self.evaluator.evaluate_qpos_batch(np.stack([rest, rest]), time_s=0.0)
        self.evaluator.evaluate_qpos_batch(np.stack([up, up]), time_s=1.0)
        self.evaluator.evaluate_qpos_batch(np.stack([up, up]), time_s=12.0)
        self.evaluator.evaluate_qpos_batch(np.stack([rest, rest]), time_s=12.6)
        self.evaluator.evaluate_qpos_batch(np.stack([rest, up]), time_s=13.0)
        first = self.evaluator.evaluate_qpos_batch(np.stack([rest, rest]), time_s=13.6)
        np.testing.assert_array_equal(first.success, [True, False])
        np.testing.assert_array_equal(first.metrics["hold_complete"], [True, True])
        np.testing.assert_array_equal(
            first.metrics["perturbations_allowed"], [False, False]
        )
        both = self.evaluator.evaluate_qpos_batch(np.stack([rest, rest]), time_s=14.6)
        np.testing.assert_array_equal(both.success, [True, True])

    def test_a_placement_within_one_control_step_still_checks_the_handles(self):
        self.hold(1, 12)
        touchdown = self.sample(12.034, REST_Z)  # lifted at the previous sample
        self.assertTrue(touchdown.metrics["handles_held"][0])
        self.assertTrue(touchdown.metrics["settling"][0])

    def test_placing_before_10_s_keeps_half_the_credit_reached_without_success(self):
        self.assertAlmostEqual(
            float(self.hold(1, 8).reward[0]), 0.63, places=4
        )  # 7 s up
        early = self.place(8.6)  # 7.5 s up: 0.675 while up, half of it kept
        self.assertEqual(self.score(early), (False, 0.3375))
        done = self.sample(9.6, REST_Z)
        self.assertTrue(done.metrics["early_placement"][0])
        self.assertEqual(self.score(done), (False, 0.3375))
        self.hold(10, 25)  # the attempt is over: lifting again changes nothing
        self.assertEqual(self.score(self.place(25.6)), (False, 0.3375))

    def test_reward_falls_after_15_s_and_a_late_placement_keeps_it(self):
        self.assertAlmostEqual(
            float(self.hold(1, 16).reward[0]), 0.9, places=4
        )  # 15 s up
        late = self.hold(16.5, 21.5)  # 20.5 s since the hold began
        self.assertAlmostEqual(float(late.reward[0]), 0.9 - 0.4 * 5.5 / 8, places=4)
        reward = float(self.place(22.1).reward[0])  # 21 s up: past the 18 s limit
        self.assertAlmostEqual(reward, 0.9 - 0.4 * 6.1 / 8, places=4)
        done = self.sample(23.1, REST_Z)
        self.assertTrue(done.metrics["late_placement"][0])
        self.assertEqual(self.score(done), (False, round(reward, 4)))
        self.assertEqual(self.score(self.sample(30, REST_Z)), (False, round(reward, 4)))

    def test_success_reward_favours_the_asked_10_to_15_s(self):
        rewards = []
        for up_until in (12, 16, 17.5, 19):  # 11.5, 15.5, 17.0, 18.5 s of air time
            self.use_evaluator()
            self.hold(1, up_until)
            self.place(up_until + 0.6)
            rewards.append(self.score(self.sample(up_until + 1.6, REST_Z)))
        expected = [
            (True, round(1.0 - 0.2 * max(a - 15, 0) / 3, 4)) for a in (11.5, 15.5, 17.0)
        ]
        self.assertEqual(rewards[:3], expected)
        self.assertFalse(rewards[3][0])  # 18.5 s is past the limit
        self.assertLess(rewards[3][1], 0.8)  # and scores below the worst success

    def test_demonstration_like_holds_succeed(self):
        # The demonstrations hold up to about 17.6 s (p95) and touch down up to about 5 s after the final drop.
        for up_until, low_until in ((18, 18.5), (12, 16.5)):
            with self.subTest(up_until=up_until, low_until=low_until):
                self.use_evaluator()
                self.hold(1, up_until)
                for t in np.arange(up_until + 0.5, low_until + 1e-9, 0.5):
                    self.sample(float(t), LOW_Z)
                touchdown = self.sample(low_until + 0.1, REST_Z)
                self.assertTrue(touchdown.metrics["settling"][0])
                done = self.sample(low_until + 1.1, REST_Z)
                self.assertTrue(done.scalar_success())
                self.assertGreaterEqual(float(done.reward[0]), 0.8)

    def test_hovering_just_above_the_table_is_a_late_placement(self):
        self.hold(1, 12)
        for t in np.arange(12.5, 20.01, 0.5):
            self.sample(
                float(t), LOW_Z
            )  # 1.6 cm up: below the lift height, not on the table
        self.sample(20.1, REST_Z)
        done = self.sample(21.1, REST_Z)
        self.assertTrue(done.metrics["late_placement"][0])
        self.assertFalse(done.scalar_success())

    def test_never_placing_decays_to_half(self):
        self.assertEqual(self.score(self.hold(1, 30, step=1.0)), (False, 0.5))

    def test_releasing_the_handles_keeps_at_most_half(self):
        for held_until, expected in ((12, 0.5), (3, 0.1125)):
            with self.subTest(held_until=held_until):
                self.use_evaluator()
                self.hold(1, held_until)
                touchdown = self.place(held_until + 0.6, held=False)
                self.assertTrue(touchdown.metrics["rough_placement"][0])
                self.assertEqual(self.score(touchdown), (False, expected))
                self.assertEqual(self.score(self.sample(30, REST_Z)), (False, expected))

    def test_ball_touched_or_lost_after_the_hold_scores_zero(self):
        for case in (
            "touched while up",
            "lost while settling",
            "lost after an early placement",
        ):
            with self.subTest(case=case):
                self.use_evaluator()
                end = 5 if case == "lost after an early placement" else 12
                self.hold(1, end)
                if case == "touched while up":
                    result = self.sample(end + 0.5, UP_Z, touch_ball=True)
                else:
                    self.assertGreater(self.place(end + 0.6).reward[0], 0.0)
                    result = self.sample(end + 1.0, REST_Z, ball_inside=False)
                self.assertTrue(result.metrics["ball_lost_after_hold"][0])
                self.assertEqual(self.score(result), (False, 0.0))
                self.assertEqual(self.score(self.sample(20, REST_Z)), (False, 0.0))

    def test_touching_the_ball_does_not_count_as_holding(self):
        self.hold(1, 5)
        self.sample(5.5, UP_Z, touch_ball=True)
        self.assertEqual(
            self.score(self.hold(6, 10)), (False, 0.36)
        )  # restarted at 6 s

    def test_flags_turn_off_each_placement_check(self):
        self.use_evaluator(require_handles_at_touchdown=False, settle_s=0.0)
        self.hold(1, 12)
        self.assertEqual(self.score(self.sample(12.1, REST_Z, held=False)), (True, 1.0))

    def test_prompts(self):
        runtime = BallTrayBalancingRuntime()
        runtime.bind(self.model, self.data)
        runtime.after_reset(
            self.model, self.data, randomization=SimpleNamespace(seed=0)
        )
        self.assertEqual(runtime.observation_prompt(), PROMPT)
        runtime.on_task_evaluation(self.sample(1, UP_Z))
        self.assertEqual(runtime.observation_prompt(), PROMPT)
        self.assertEqual(get_task_spec("ball_tray_balancing").prompt, PROMPT)
        self.use_evaluator(hold_window_s=None)
        runtime.on_task_evaluation(self.sample(0, REST_Z))
        self.assertEqual(
            runtime.observation_prompt(),
            "balance the ball in the tray for at least 7 seconds",
        )


if __name__ == "__main__":
    unittest.main()
