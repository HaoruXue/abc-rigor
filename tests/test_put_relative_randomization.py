"""Put-relative placement searches unused slots while retaining strict clearance."""

import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from abc_sim.randomization.tasks import put_relative as module


class OrderedRng:
    """Keep a fixed category/slot order and place every candidate at its center."""

    def integers(self, low, high=None):
        return low if high is not None else 0

    def choice(self, values, *, size, replace):
        return np.asarray(values[:size])

    def shuffle(self, values):
        pass

    def uniform(self, low, high):
        return 0.0


class PutRelativePlacementTest(unittest.TestCase):
    def select(self, slots, radii, *, rng=None, metric_error=None):
        categories = tuple(f"category_{index}" for index in range(len(radii)))
        metrics = {
            category: {"spawn_z": 0.8, "footprint_radius": radius}
            for category, radius in zip(categories, radii)
        }
        with ExitStack() as stack:
            for name, value in (
                ("_PUT_RELATIVE_OBJECT_COUNT_RANGE", (len(radii), len(radii))),
                ("_PUT_RELATIVE_OBJECT_CATEGORIES", categories),
                ("_PUT_RELATIVE_SLOTS", slots),
                ("_PUT_RELATIVE_MAX_LAYOUT_TRIES", 3),
            ):
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(
                patch.object(
                    module.PutRelativeRandomizer,
                    "_get_variants",
                    lambda self, category: [Path(category)],
                )
            )
            stack.enter_context(
                patch.object(
                    module,
                    "_put_relative_variant_metrics",
                    side_effect=metric_error or (lambda path: metrics[path.name]),
                )
            )
            return module.PutRelativeRandomizer()._sample_selections(rng or OrderedRng())

    def test_searches_another_unused_slot_when_assigned_slot_cannot_fit(self):
        selections = self.select(((0.0, 0.0), (0.1, 0.0), (1.0, 0.0)), (0.06, 0.06))
        self.assertEqual([(row["x"], row["y"]) for row in selections], [(0.0, 0.0), (1.0, 0.0)])
        self.assertTrue(module._put_relative_xy_clearance_ok(selections[1], selections[:1], clearance=0.015))

    def test_preferred_feasible_slots_are_retained(self):
        selections = self.select(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), (0.05, 0.05))
        self.assertEqual([row["x"] for row in selections], [0.0, 1.0])

    def test_skipped_slot_remains_available_to_a_smaller_object(self):
        selections = self.select(((0.0, 0.0), (0.1, 0.0), (1.0, 0.0)), (0.02, 0.1, 0.02))
        self.assertEqual([row["x"] for row in selections], [0.0, 1.0, 0.1])

    def test_exhausted_slots_still_raise(self):
        with self.assertRaisesRegex(RuntimeError, "could not place non-overlapping objects"):
            self.select(((0.0, 0.0), (0.1, 0.0)), (0.1, 0.1))

    def test_invalid_asset_errors_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "bad asset"):
            self.select(((0.0, 0.0), (1.0, 0.0)), (0.05, 0.05), metric_error=RuntimeError("bad asset"))

    def test_real_generator_repeats_exactly(self):
        arguments = (((0.0, 0.0), (0.1, 0.0), (1.0, 0.0)), (0.06, 0.06))
        first = self.select(*arguments, rng=np.random.default_rng(19))
        second = self.select(*arguments, rng=np.random.default_rng(19))
        self.assertEqual(first, second)
        self.assertTrue(module._put_relative_xy_clearance_ok(first[1], first[:1], clearance=0.015))


if __name__ == "__main__":
    unittest.main()
