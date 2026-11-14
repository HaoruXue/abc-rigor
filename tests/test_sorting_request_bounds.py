"""Sorting request maxima select the complete declared physical candidate pools."""

import unittest

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:
    if exc.name != "mujoco":
        raise
    mujoco = None

if mujoco is not None:
    from abc_sim.randomization.requests import (
        LegoBlocksResetRequest,
        NutsBoltsResetRequest,
    )
    from abc_sim.randomization.tasks.sorting import (
        LegoBlocksSortingRandomizer,
        NutsBoltsSortingRandomizer,
    )


@unittest.skipIf(mujoco is None, "The native randomization package requires MuJoCo.")
class SortingRequestBoundsTest(unittest.TestCase):
    def test_nuts_bolts_capacity_requests_select_every_candidate(self):
        randomizer = NutsBoltsSortingRandomizer()
        capacities = {
            prefix: sum(
                name.startswith(prefix + "_")
                for name in randomizer.candidate_object_body_names
            )
            for prefix in ("nut", "bolt")
        }
        request = NutsBoltsResetRequest(
            nut_count=capacities["nut"], bolt_count=capacities["bolt"]
        )
        selected, metadata = randomizer._sample_active_object_body_names(
            np.random.default_rng(41), request
        )
        self.assertEqual(set(selected), set(randomizer.candidate_object_body_names))
        self.assertEqual(
            metadata["active_counts"],
            {"nuts": capacities["nut"], "bolts": capacities["bolt"]},
        )
        for prefix in ("nut", "bolt"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                NutsBoltsResetRequest(**{f"{prefix}_count": capacities[prefix] + 1})

    def test_lego_capacity_request_selects_every_color_candidate(self):
        randomizer = LegoBlocksSortingRandomizer()
        capacities = {
            color: sum(
                name.startswith(color + "_")
                for name in randomizer.candidate_object_body_names
            )
            for color in ("red", "yellow", "blue")
        }
        self.assertEqual(len(set(capacities.values())), 1)
        capacity = capacities["red"]
        selected, metadata = randomizer._sample_active_object_body_names(
            np.random.default_rng(41), LegoBlocksResetRequest(blocks_per_color=capacity)
        )
        self.assertEqual(set(selected), set(randomizer.candidate_object_body_names))
        self.assertEqual(metadata["active_counts"], capacities)
        with self.assertRaises(ValueError):
            LegoBlocksResetRequest(blocks_per_color=capacity + 1)


if __name__ == "__main__":
    unittest.main()
