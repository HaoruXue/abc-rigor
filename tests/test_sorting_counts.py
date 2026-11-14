"""Explicit sorting counts preserve scene membership and native partial rewards."""

import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:
    if exc.name != "mujoco":
        raise
    mujoco = None

if mujoco is not None:
    from abc_sim.randomization.core import RandomizationState, SceneRandomizer
    from abc_sim.randomization.requests import LegoBlocksResetRequest, NutsBoltsResetRequest
    from abc_sim.randomization.tasks.sorting import LegoBlocksSortingRandomizer, NutsBoltsSortingRandomizer
    from abc_sim.task_eval.lego_blocks import LegoBlocksSortingEvaluator
    from abc_sim.task_eval.nuts_bolts import NutsBoltsSortingEvaluator
    from abc_sim.task_specs import get_task_spec


def sorting_model(groups):
    bodies = []
    for x, (bin_name, site_name, names) in enumerate(groups):
        bodies.append(
            f'<body name="{bin_name}" pos="{x} 0 0">'
            f'<freejoint name="{bin_name}_joint"/>'
            '<geom type="box" size=".2 .2 .1" mass="1"/>'
            f'<site name="{site_name}" type="box" size=".2 .2 .2"/></body>'
        )
        for name in names:
            bodies.append(
                f'<body name="{name}" pos="{x} 0 0">'
                f'<freejoint name="{name}_joint"/>'
                '<geom type="sphere" size=".01" mass=".01"/></body>'
            )
    return mujoco.MjModel.from_xml_string("<mujoco><worldbody>" + "".join(bodies) + "</worldbody></mujoco>")


def parked_rows(model, candidates, active_by_world):
    rows = np.repeat(mujoco.MjData(model).qpos[None], len(active_by_world), axis=0)
    for world, active in enumerate(active_by_world):
        for joint in set(candidates) - set(active):
            rows[world, model.joint(joint).qposadr[0] + 2] = -1
    return rows


@unittest.skipIf(mujoco is None, "MuJoCo is required for native sorting evaluation.")
class SortingCountsTest(unittest.TestCase):
    def test_explicit_nuts_bolts_counts_are_repeatable_and_unique(self):
        request = NutsBoltsResetRequest(nut_count=3, bolt_count=3)
        randomizer = NutsBoltsSortingRandomizer()
        names, metadata = randomizer._sample_active_object_body_names(np.random.default_rng(17), request)
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertEqual(metadata["active_counts"], {"nuts": 3, "bolts": 3})
        self.assertEqual(
            (names, metadata), randomizer._sample_active_object_body_names(np.random.default_rng(17), request)
        )
        self.assertTrue(set(names) <= set(randomizer.candidate_object_body_names))

    def test_explicit_lego_count_binds_every_color(self):
        randomizer = LegoBlocksSortingRandomizer()
        names, metadata = randomizer._sample_active_object_body_names(
            np.random.default_rng(17), LegoBlocksResetRequest(blocks_per_color=2)
        )
        self.assertEqual(metadata["active_counts"], {"red": 2, "yellow": 2, "blue": 2})
        self.assertEqual(len(set(names)), 6)
        randomizer._configure_active_scene(names)
        root = ET.fromstring(randomizer._base_scene_xml_string)
        remaining = {body.get("name") for body in root.iter("body")}
        self.assertEqual(remaining & set(randomizer.candidate_object_body_names), set(names))
        self.assertEqual(randomizer._parked_joints, ())

    def test_original_default_count_ranges_are_unchanged(self):
        nuts = NutsBoltsSortingRandomizer()
        lego = LegoBlocksSortingRandomizer()
        for seed in range(20):
            rng = np.random.default_rng(seed)
            expected_nuts, expected_bolts = int(rng.integers(8, 21)), int(rng.integers(8, 21))
            _, counts = nuts._sample_active_object_body_names(np.random.default_rng(seed))
            self.assertEqual((counts["nut_count"], counts["bolt_count"]), (expected_nuts, expected_bolts))
            _, counts = lego._sample_active_object_body_names(np.random.default_rng(seed))
            self.assertTrue(all(3 <= value <= 9 for value in counts["active_counts"].values()))

    def test_invalid_or_misrouted_counts_fail_before_sampling(self):
        for value in (0, 21, 3.5, "3", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NutsBoltsResetRequest.from_value({"nut_count": value})
        for value in (0, 10, 2.5, "2", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LegoBlocksResetRequest.from_value({"blocks_per_color": value})
        with self.assertRaises(ValueError):
            LegoBlocksResetRequest.from_value({"nut_count": 3})
        with self.assertRaises(TypeError):
            NutsBoltsResetRequest.from_value(LegoBlocksResetRequest(blocks_per_color=2))

    def test_randomize_routes_typed_counts_and_scene_flags(self):
        for randomizer, request, expected in (
            (
                NutsBoltsSortingRandomizer(),
                NutsBoltsResetRequest(nut_count=3, bolt_count=3, randomize_scales=False, park_inactive=True),
                {"nuts": 3, "bolts": 3},
            ),
            (
                LegoBlocksSortingRandomizer(),
                LegoBlocksResetRequest(blocks_per_color=2, randomize_scales=False, park_inactive=True),
                {"red": 2, "yellow": 2, "blue": 2},
            ),
        ):
            randomizer._env_ref = SimpleNamespace(model=object(), data=object())
            with (
                patch.object(randomizer, "_configure_active_scene") as configure,
                patch.object(randomizer, "_reload_configured_scene"),
                patch.object(SceneRandomizer, "randomize", return_value=RandomizationState(17, {})) as native,
            ):
                result = randomizer.randomize(object(), object(), seed=17, request=request)
            self.assertEqual(result.metadata["active_counts"], expected)
            self.assertEqual(result.metadata["active_object_count"], 6)
            self.assertEqual(len(result.metadata["active_object_joints"]), 6)
            self.assertTrue(configure.call_args.kwargs["park_inactive"])
            self.assertFalse(native.call_args.kwargs["request"]["randomize_scales"])

    def test_lego_native_count_metadata_and_partial_reward(self):
        model = sorting_model(
            [
                (f"{color}_lego_sorting_bin", f"{color}_lego_bin_goal_region", [f"{color}_lego_1", f"{color}_lego_2"])
                for color in ("red", "yellow", "blue")
            ]
        )
        evaluator = LegoBlocksSortingEvaluator(model=model, spec=get_task_spec("sort_lego_blocks"))
        data = mujoco.MjData(model)
        complete = evaluator.evaluate_qpos_batch(data.qpos[None]).to_info(squeeze=True)
        self.assertEqual((complete["active_object_count"], complete["num_correct"], complete["reward"]), (6, 6, 1))
        data.qpos[model.joint("blue_lego_2_joint").qposadr[0]] = 5
        partial = evaluator.evaluate_qpos_batch(data.qpos[None]).to_info(squeeze=True)
        self.assertEqual((partial["active_object_count"], partial["num_correct"]), (6, 5))
        self.assertEqual(partial["reward"], float(np.float32(5 / 6)))
        self.assertFalse(partial["success"])

    def test_nuts_native_orientation_and_count_metadata(self):
        model = sorting_model(
            [
                ("nuts_sorting_bin", "nuts_bin_goal_region", [f"nut_{i}" for i in range(1, 4)]),
                ("bolts_sorting_bin", "bolts_bin_goal_region", [f"bolt_{i}" for i in range(1, 4)]),
            ]
        )
        evaluator = NutsBoltsSortingEvaluator(model=model, spec=get_task_spec("sort_nuts_and_bolts"))
        data = mujoco.MjData(model)
        complete = evaluator.evaluate_qpos_batch(data.qpos[None]).to_info(squeeze=True)
        self.assertEqual((complete["active_object_count"], complete["num_correct"], complete["reward"]), (6, 6, 1))
        self.assertEqual(complete["sorted_orientation"], "spec")
        for prefix, x in (("nut", 1), ("bolt", 0)):
            for index in range(1, 4):
                data.qpos[model.joint(f"{prefix}_{index}_joint").qposadr[0]] = x
        inverted = evaluator.evaluate_qpos_batch(data.qpos[None]).to_info(squeeze=True)
        self.assertEqual(inverted["num_correct_inverted"], 6)
        self.assertEqual(inverted["sorted_orientation"], "inverted")
        data.qpos[model.joint("nut_3_joint").qposadr[0]] = 5
        partial = evaluator.evaluate_qpos_batch(data.qpos[None]).to_info(squeeze=True)
        self.assertEqual((partial["active_object_count"], partial["num_correct"]), (6, 5))
        self.assertEqual(partial["reward"], float(np.float32(5 / 6)))

    def test_lego_parked_candidates_respect_each_world_mask_and_reset(self):
        colors = ("red", "yellow", "blue")
        model = sorting_model(
            [
                (
                    f"{color}_lego_sorting_bin",
                    f"{color}_lego_bin_goal_region",
                    [f"{color}_lego_{index}" for index in (1, 2, 3)],
                )
                for color in colors
            ]
        )
        evaluator = LegoBlocksSortingEvaluator(model=model, spec=get_task_spec("sort_lego_blocks"))
        first = [
            [f"{color}_lego_{index}_joint" for color in colors for index in (1, 2)],
            [f"{color}_lego_3_joint" for color in colors],
        ]
        candidates = first[0] + first[1]
        evaluator.reset(nworld=2)
        evaluator.set_active_object_joints(first)
        rows = parked_rows(model, candidates, first)
        rows[0, model.joint("blue_lego_2_joint").qposadr[0]] = 5
        result = evaluator.evaluate_qpos_batch(rows)
        np.testing.assert_array_equal(result.metrics["active_object_count"], [6, 3])
        np.testing.assert_array_equal(result.metrics["num_correct"], [5, 3])
        np.testing.assert_allclose(result.reward, [5 / 6, 1])
        np.testing.assert_array_equal(result.success, [False, True])

        second = [first[1], first[0]]
        evaluator.reset(nworld=2)
        evaluator.set_active_object_joints(second)
        rows = parked_rows(model, candidates, second)
        rows[0, model.joint("yellow_lego_3_joint").qposadr[0]] = 0
        rows[1, model.joint("blue_lego_1_joint").qposadr[0]] = 5
        result = evaluator.evaluate_qpos_batch(rows)
        np.testing.assert_array_equal(result.metrics["active_object_count"], [3, 6])
        np.testing.assert_array_equal(result.metrics["num_correct"], [2, 5])
        np.testing.assert_array_equal(result.metrics["num_wrong_bin"], [1, 0])
        np.testing.assert_allclose(result.reward, [2 / 3, 5 / 6])
        np.testing.assert_array_equal(result.success, [False, False])
        np.testing.assert_array_equal(result.metrics["ever_success"], [False, False])

    def test_nuts_parked_candidates_respect_world_masks_orientations_and_reset(self):
        model = sorting_model(
            [
                ("nuts_sorting_bin", "nuts_bin_goal_region", [f"nut_{i}" for i in range(1, 5)]),
                ("bolts_sorting_bin", "bolts_bin_goal_region", [f"bolt_{i}" for i in range(1, 5)]),
            ]
        )
        evaluator = NutsBoltsSortingEvaluator(model=model, spec=get_task_spec("sort_nuts_and_bolts"))
        first = [
            [f"{prefix}_{index}_joint" for prefix in ("nut", "bolt") for index in (1, 2, 3)],
            ["nut_4_joint", "bolt_4_joint"],
        ]
        candidates = first[0] + first[1]
        evaluator.reset(nworld=2)
        evaluator.set_active_object_joints(first)
        rows = parked_rows(model, candidates, first)
        rows[0, model.joint("nut_3_joint").qposadr[0]] = 5
        rows[1, model.joint("nut_4_joint").qposadr[0]] = 1
        rows[1, model.joint("bolt_4_joint").qposadr[0]] = 0
        result = evaluator.evaluate_qpos_batch(rows)
        np.testing.assert_array_equal(result.metrics["active_object_count"], [6, 2])
        np.testing.assert_array_equal(result.metrics["num_correct_spec"], [5, 0])
        np.testing.assert_array_equal(result.metrics["num_correct_inverted"], [0, 2])
        np.testing.assert_allclose(result.reward, [5 / 6, 1])
        self.assertEqual(result.metrics["sorted_orientation"], ["none", "inverted"])

        second = [
            ["nut_4_joint", "bolt_4_joint"],
            ["nut_1_joint", "nut_2_joint", "bolt_1_joint", "bolt_2_joint", "bolt_3_joint"],
        ]
        evaluator.reset(nworld=2)
        evaluator.set_active_object_joints(second)
        rows = parked_rows(model, candidates, second)
        rows[0, model.joint("bolt_4_joint").qposadr[0]] = 5
        for joint in second[1]:
            rows[1, model.joint(joint).qposadr[0]] = 1 if joint.startswith("nut") else 0
        rows[1, model.joint("nut_2_joint").qposadr[0]] = 5
        result = evaluator.evaluate_qpos_batch(rows)
        np.testing.assert_array_equal(result.metrics["active_object_count"], [2, 5])
        np.testing.assert_array_equal(result.metrics["num_correct_spec"], [1, 0])
        np.testing.assert_array_equal(result.metrics["num_correct_inverted"], [0, 4])
        np.testing.assert_array_equal(result.metrics["num_correct"], [1, 4])
        np.testing.assert_allclose(result.reward, [1 / 2, 4 / 5])
        np.testing.assert_array_equal(result.success, [False, False])
        np.testing.assert_array_equal(result.metrics["ever_success"], [False, False])


if __name__ == "__main__":
    unittest.main()
