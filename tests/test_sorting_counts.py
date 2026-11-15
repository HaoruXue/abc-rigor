"""Optional sorting counts and native scores for each world's active objects."""

import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from abc_sim.randomization.core import RandomizationState, SceneRandomizer
from abc_sim.randomization.requests import LegoBlocksResetRequest, NutsBoltsResetRequest
from abc_sim.randomization.tasks.sorting import LegoBlocksSortingRandomizer, NutsBoltsSortingRandomizer
from abc_sim.task_eval.lego_blocks import LegoBlocksSortingEvaluator
from abc_sim.task_eval.nuts_bolts import NutsBoltsSortingEvaluator
from abc_sim.task_specs import get_task_spec


class SortingCountsTest(unittest.TestCase):
    def test_requested_counts_capacity_repeatability_and_scene_membership(self):
        cases = (
            (NutsBoltsSortingRandomizer, NutsBoltsResetRequest(nut_count=1, bolt_count=20), {"nuts": 1, "bolts": 20}),
            (NutsBoltsSortingRandomizer, NutsBoltsResetRequest(nut_count=3, bolt_count=3), {"nuts": 3, "bolts": 3}),
            (LegoBlocksSortingRandomizer, LegoBlocksResetRequest(blocks_per_color=1), dict.fromkeys(("red", "yellow", "blue"), 1)),
            (LegoBlocksSortingRandomizer, LegoBlocksResetRequest(blocks_per_color=9), dict.fromkeys(("red", "yellow", "blue"), 9)),
        )
        for cls, request, expected in cases:
            with self.subTest(request=request):
                randomizer = cls()
                names, metadata = randomizer._sample_active_object_body_names(np.random.default_rng(17), request)
                self.assertEqual(metadata["active_counts"], expected)
                self.assertEqual(len(names), len(set(names)))
                self.assertEqual(len(names), sum(expected.values()))
                self.assertEqual((names, metadata), randomizer._sample_active_object_body_names(np.random.default_rng(17), request))
                randomizer._configure_active_scene(names)
                bodies = {body.get("name") for body in ET.fromstring(randomizer._base_scene_xml_string).iter("body")}
                self.assertEqual(bodies & set(randomizer.candidate_object_body_names), set(names))

    def test_omitted_counts_keep_default_ranges_and_partial_requests(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            expected = (int(rng.integers(8, 21)), int(rng.integers(8, 21)))
            _, counts = NutsBoltsSortingRandomizer()._sample_active_object_body_names(np.random.default_rng(seed))
            self.assertEqual((counts["nut_count"], counts["bolt_count"]), expected)
            _, counts = LegoBlocksSortingRandomizer()._sample_active_object_body_names(np.random.default_rng(seed))
            self.assertTrue(all(3 <= n <= 9 for n in counts["active_counts"].values()))
        for field in ("nut_count", "bolt_count"):
            _, counts = NutsBoltsSortingRandomizer()._sample_active_object_body_names(
                np.random.default_rng(17), NutsBoltsResetRequest.from_value({field: 1})
            )
            self.assertEqual(counts[field], 1)
            self.assertTrue(8 <= counts["bolt_count" if field == "nut_count" else "nut_count"] <= 20)

    def test_invalid_counts_and_wrong_task_requests_fail(self):
        for cls, field, maximum in (
            (NutsBoltsResetRequest, "nut_count", 20),
            (NutsBoltsResetRequest, "bolt_count", 20),
            (LegoBlocksResetRequest, "blocks_per_color", 9),
        ):
            for value in (0, maximum + 1, 1.5, "2", True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    cls.from_value({field: value})
        with self.assertRaises(ValueError):
            LegoBlocksResetRequest.from_value({"nut_count": 2})
        with self.assertRaises(TypeError):
            NutsBoltsResetRequest.from_value(LegoBlocksResetRequest(blocks_per_color=2))

    def test_randomize_keeps_existing_scene_controls(self):
        for cls, counts in (
            (NutsBoltsSortingRandomizer, {"nut_count": 3, "bolt_count": 3}),
            (LegoBlocksSortingRandomizer, {"blocks_per_color": 2}),
        ):
            for scale_flag in (False, None, 0):
                with self.subTest(task=cls.__name__, scale_flag=scale_flag):
                    randomizer = cls()
                    randomizer._env_ref = SimpleNamespace(model=object(), data=object())
                    request = {**counts, "park_inactive": 1, "randomize_scales": scale_flag, "bin_visual_style": ""}
                    with (
                        patch.object(randomizer, "_configure_active_scene") as configure,
                        patch.object(randomizer, "_reload_configured_scene") as reload_scene,
                        patch.object(SceneRandomizer, "randomize", return_value=RandomizationState(17, {})) as native,
                    ):
                        state = randomizer.randomize(object(), object(), seed=17, request=request)
                    self.assertEqual(state.metadata["active_object_count"], 6)
                    self.assertEqual(len(state.metadata["active_object_joints"]), 6)
                    self.assertTrue(configure.call_args.kwargs["park_inactive"])
                    self.assertIs(native.call_args.kwargs["request"]["randomize_scales"], scale_flag)
                    reload_scene.assert_called_once()

    def test_native_active_totals_rewards_world_masks_orientation_and_reset(self):
        for lego in (False, True):
            with self.subTest(lego=lego):
                prefixes = ("red_lego", "yellow_lego", "blue_lego") if lego else ("nut", "bolt")
                bins = tuple(f"{color}_lego_sorting_bin" for color in ("red", "yellow", "blue")) if lego else ("nuts_sorting_bin", "bolts_sorting_bin")
                sites = tuple(f"{color}_lego_bin_goal_region" for color in ("red", "yellow", "blue")) if lego else ("nuts_bin_goal_region", "bolts_bin_goal_region")
                bodies = []
                for x, (prefix, bin_name, site) in enumerate(zip(prefixes, bins, sites)):
                    bodies.append(
                        f'<body name="{bin_name}" pos="{x} 0 0"><freejoint name="{bin_name}_joint"/>'
                        f'<geom type="sphere" size=".01" mass="1"/><site name="{site}" type="box" size=".2 .2 .2"/></body>'
                    )
                    bodies.extend(
                        f'<body name="{prefix}_{i}" pos="{x} 0 0"><freejoint name="{prefix}_{i}_joint"/>'
                        '<geom type="sphere" size=".01" mass=".01"/></body>' for i in (1, 2)
                    )
                model = mujoco.MjModel.from_xml_string("<mujoco><worldbody>" + "".join(bodies) + "</worldbody></mujoco>")
                cls = LegoBlocksSortingEvaluator if lego else NutsBoltsSortingEvaluator
                evaluator = cls(model=model, spec=get_task_spec("lego blocks sorting" if lego else "nuts and bolts sorting"))
                candidates = [f"{prefix}_{i}_joint" for prefix in prefixes for i in (1, 2)]
                active = [[f"{prefix}_1_joint" for prefix in prefixes], candidates]
                for reset in (False, True):
                    if reset:
                        active.reverse()
                    evaluator.reset(nworld=2)
                    evaluator.set_active_object_joints(active)
                    rows = np.repeat(mujoco.MjData(model).qpos[None], 2, axis=0)
                    for world, joints in enumerate(active):
                        for joint in set(candidates) - set(joints):
                            rows[world, model.joint(joint).qposadr[0] + 2] = -1
                    if not lego and not reset:
                        for joint in active[1]:
                            rows[1, model.joint(joint).qposadr[0]] = 1 if joint.startswith("nut") else 0
                    rows[0, model.joint(active[0][0]).qposadr[0]] = 1 if lego and reset else 5
                    if reset:
                        rows[1, model.joint(active[1][0]).qposadr[0]] = 5
                    result = evaluator.evaluate_qpos_batch(rows)
                    totals = np.array([len(joints) for joints in active])
                    correct = totals - [1, int(reset)]
                    np.testing.assert_array_equal(result.metrics["active_object_count"], totals)
                    np.testing.assert_array_equal(result.metrics["num_correct"], correct)
                    wrong = np.array([int(lego and reset), 0])
                    np.testing.assert_allclose(result.reward, (correct - wrong) / totals)
                    np.testing.assert_array_equal(result.success, [False, not reset])
                    np.testing.assert_array_equal(result.metrics["ever_success"], [False, not reset])
                    if lego:
                        np.testing.assert_array_equal(result.metrics["num_wrong_bin"], wrong)
                    else:
                        self.assertEqual(result.metrics["sorted_orientation"], ["none", "none" if reset else "inverted"])


if __name__ == "__main__":
    unittest.main()
