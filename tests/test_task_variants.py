"""Versioned selection, collision handling and native-task preservation."""

import json
import unittest
from dataclasses import asdict, replace

import abc_sim
from abc_sim.task_specs import (
    SimTaskSpec,
    TaskSpecRegistry,
    get_task_spec,
    list_task_specs,
)
from abc_sim.task_variants import (
    BottleThroughputTask,
    LegoThroughputTask,
    NutsBoltsThroughputTask,
    free_placement_spelling_task,
)


class TaskVariantTests(unittest.TestCase):
    def test_public_registration_keeps_base_catalogue_and_shared_prompt(self):
        base = get_task_spec("put_plastic_bottles_in_bin")
        catalogue = list_task_specs()
        variant = BottleThroughputTask().task_spec()
        self.assertIs(abc_sim.register_task_spec(variant), variant)
        self.assertEqual(abc_sim.register_task_spec(variant), variant)
        self.assertEqual(abc_sim.select_task_spec(variant.name, version="v1"), variant)
        self.assertIs(get_task_spec(base.prompt), base)
        self.assertIs(list_task_specs(), catalogue)
        self.assertIn(variant, abc_sim.list_task_variants())
        for name, version in (
            (variant.name, "v2"),
            (base.prompt, "v1"),
            (variant.name.upper(), "v1"),
        ):
            with (
                self.subTest(name=name, version=version),
                self.assertRaises(ValueError),
            ):
                abc_sim.select_task_spec(name, version=version)

    def test_registration_rejects_every_normalized_key_collision_atomically(self):
        registry = TaskSpecRegistry(list_task_specs())
        original = get_task_spec("put_plastic_bottles_in_bin")
        good = BottleThroughputTask().task_spec()
        defects = (
            replace(good, name="PUT--plastic bottles IN_bin"),
            replace(good, aliases=("throw plastic bottles in bin",)),
            replace(good, prompt="SORT--LEGO blocks"),
            replace(good, aliases=(original.prompt,)),
            replace(good, name=original.prompt),
            replace(good, aliases=("   ",)),
            replace(good, prompt=""),
        )
        for bad in defects:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                registry.register(bad)
            with self.assertRaises(KeyError):
                registry.get(good.name)
            self.assertIs(registry.get(original.prompt), original)
        registry.register(good)
        with self.assertRaises(ValueError):
            registry.register(replace(good, reset_request=(("bottle_count", 2),)))
        self.assertEqual(registry.get(good.name), good)

    def test_base_registration_checks_prompt_and_alias_collisions(self):
        registry = TaskSpecRegistry(list_task_specs())
        for fields in (
            {"name": "new", "prompt": "put-the-plastic-bottles-in-the-bin"},
            {"name": "new", "prompt": "new prompt", "aliases": ("sort-LEGO-blocks",)},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                registry.register(SimTaskSpec(env_task="blocks", **fields))
            with self.assertRaises(KeyError):
                registry.get("new")

    def test_variant_requires_canonical_existing_base_version_and_finite_horizon(self):
        variant = BottleThroughputTask().task_spec()
        for fields in (
            {"version": None},
            {"base_task": None},
            {"version": ""},
            {"horizon_seconds": 0},
            {"horizon_seconds": float("nan")},
            {"horizon_seconds": True},
            {"horizon_seconds": "30"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(variant, **fields)
        registry = TaskSpecRegistry(list_task_specs())
        with self.assertRaises(ValueError):
            registry.register(replace(variant, base_task="put_bottles"))
        with self.assertRaises(KeyError):
            registry.register(replace(variant, base_task="absent"))

    def test_all_existing_count_and_horizon_identities(self):
        conditions = [
            *(
                BottleThroughputTask(count, seconds)
                for count in range(2, 7)
                for seconds in (30, 60)
            ),
            *(
                LegoThroughputTask(count, seconds)
                for count in range(1, 10)
                for seconds in (30, 60)
            ),
            *(
                NutsBoltsThroughputTask(nuts, bolts, seconds)
                for nuts in range(1, 21)
                for bolts in range(1, 21)
                for seconds in (30, 60)
            ),
        ]
        self.assertEqual(len({condition.task_id for condition in conditions}), 828)
        registry = TaskSpecRegistry(list_task_specs())
        for condition in conditions:
            spec = condition.task_spec()
            base = get_task_spec(spec.base_task)
            self.assertEqual(spec.prompt, base.prompt)
            self.assertEqual(spec.env_task, base.env_task)
            self.assertEqual(spec.evaluator_name, base.evaluator_name)
            self.assertEqual(spec.evaluator_options, base.evaluator_options)
            self.assertEqual(spec.horizon_seconds, condition.horizon_seconds)
            self.assertEqual(spec.name, condition.task_id)
            self.assertEqual(spec.version, "v1")
            self.assertEqual(
                json.loads(json.dumps(asdict(condition))), asdict(condition)
            )
            restored = type(condition)(**json.loads(json.dumps(asdict(condition))))
            self.assertEqual(restored.task_spec(), spec)
            registry.register(spec)
            self.assertIs(registry.get(base.prompt), base)

    def test_invalid_counts_and_horizons_do_not_coerce(self):
        for constructor, field, values in (
            (BottleThroughputTask, "bottle_count", (1, 7, True, 2.5, "6")),
            (LegoThroughputTask, "blocks_per_color", (0, 10, True, 2.5, "2")),
            (NutsBoltsThroughputTask, "nut_count", (0, 21, True, 2.5, "3")),
            (NutsBoltsThroughputTask, "bolt_count", (0, 21, True, 2.5, "3")),
        ):
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    constructor(**{field: value})
        for constructor in (
            BottleThroughputTask,
            LegoThroughputTask,
            NutsBoltsThroughputTask,
        ):
            for value in (0, 120, True, 30.0, "30"):
                with (
                    self.subTest(constructor=constructor, value=value),
                    self.assertRaises(ValueError),
                ):
                    constructor(horizon_seconds=value)

    def test_shipped_reset_evidence_and_native_progress_fields(self):
        cases = (
            (
                BottleThroughputTask(),
                {"bottle_count": 6},
                {"num_bottles_in_bin": 2, "num_active_bottles": 6},
                {},
            ),
            (
                LegoThroughputTask(),
                {"active_counts": {"red": 2, "yellow": 2, "blue": 2}},
                {"num_correct": 2, "active_object_count": 6},
                {},
            ),
            (
                NutsBoltsThroughputTask(),
                {"active_counts": {"nuts": 3, "bolts": 3}},
                {
                    "num_correct": 2,
                    "active_object_count": 6,
                    "num_correct_spec": 1,
                    "num_correct_inverted": 2,
                },
                {"num_correct_spec": 1, "num_correct_inverted": 2},
            ),
        )
        for condition, metadata, metrics, details in cases:
            condition.validate_reset({"metadata": metadata})
            self.assertEqual(condition.progress_counts(metrics), (2, 6, details))
            with self.assertRaises((KeyError, ValueError)):
                condition.validate_reset(
                    {"metadata": {"bottle_count": 2, "active_counts": {}}}
                )

    def test_versioned_reset_cannot_override_fixed_fields(self):
        for condition in (
            BottleThroughputTask(),
            LegoThroughputTask(),
            NutsBoltsThroughputTask(),
        ):
            spec = condition.task_spec()
            expected = spec.reset_request_options()
            self.assertEqual(spec.resolve_reset_request(None), expected)
            self.assertEqual(spec.resolve_reset_request({}), expected)
            self.assertEqual(
                spec.resolve_reset_request({"randomize_scales": False}),
                {**expected, "randomize_scales": False},
            )
            for name, value in expected.items():
                for wrong in (value + 1, float(value), str(value), True):
                    with (
                        self.subTest(name=name, wrong=wrong),
                        self.assertRaises(ValueError),
                    ):
                        spec.resolve_reset_request({name: wrong})
        original = get_task_spec("count_four_into_opaque_box")
        self.assertEqual(
            original.resolve_reset_request({"target_count": 2}), {"target_count": 2}
        )

    def test_spelling_uses_new_identity_and_keeps_original_goal_and_score(self):
        before = tuple(list_task_specs())
        spec = free_placement_spelling_task()
        original = get_task_spec("spell_random_2")
        self.assertEqual(spec.name, "spell_random_2_free_placement_60s_v1")
        self.assertEqual(spec.horizon_seconds, 60)
        self.assertEqual(spec.evaluator_name, "abc-spelling-free-placement-v1")
        self.assertEqual(spec.reset_request, original.reset_request)
        self.assertEqual(spec.prompt, original.prompt)
        registry = TaskSpecRegistry(before)
        registry.register(spec)
        self.assertIs(registry.get(original.prompt), original)
        self.assertEqual(list_task_specs(), before)
        for length in (2, 3, 4):
            self.assertEqual(
                get_task_spec(f"spell_random_{length}").evaluator_name, "spell_word"
            )
        for word in ("cat", "dog", "fish", "bair", "xdof", "abc", "yam", "agi"):
            self.assertEqual(
                get_task_spec(f"spell_{word}").evaluator_kwargs(), {"word": word}
            )


if __name__ == "__main__":
    unittest.main()
