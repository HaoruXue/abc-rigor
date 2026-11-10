"""Seeded dish-rack placement with captured geometry and native MuJoCo checks."""

import hashlib
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
from abc_sim.randomization.assets import dishrack as assets
from abc_sim.randomization.core import RandomizationSamplingError
from abc_sim.randomization.tasks import dishrack as task

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dishrack_seed_3063706685.json").read_text())


def geometry_model():
    """Represent recorded footprints with primitives, without mesh downloads."""
    selected = FIXTURE["selected"]
    bodies = []
    for name, (position, quaternion) in selected["nominals"].items():
        if name == "dishrack":
            half_x, half_y = selected["rack_half_extents_xy"]
            geom = f'<geom type="box" size="{half_x} {half_y} .01" mass="1"/>'
        else:
            index = 0 if name == "plate_joint" else int(name.rsplit("_", 1)[1])
            radius = selected["plate_radii"][index]
            geom = f'<geom type="cylinder" size="{radius} .005" mass=".1"/>'
        pos = " ".join(map(str, position))
        quat = " ".join(map(str, quaternion))
        bodies.append(f'<body pos="{pos}" quat="{quat}"><freejoint name="{name}"/>{geom}</body>')
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody>" + "".join(bodies) + "</worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class GeometryDishRack(task.DishRackRandomizer):
    """Replace asset loading while retaining the real seeded placement search."""

    def __init__(self, model, data):
        super().__init__()
        self._env_ref = SimpleNamespace(model=model, data=data)
        self.proposals = []
        self.contact_checks = 0

    def _reload_variant_scene(self, rack, plates, scales):
        selected = FIXTURE["selected"]
        if (
            rack != selected["variants"]["dish_rack"]
            or plates != selected["plate_variants"]
            or scales != selected["scales"]
        ):
            raise AssertionError("The seeded asset selection differs from the fixture.")
        self._current_variant_names = {"dish_rack": rack, "plate": plates[0]}
        self._current_plate_variants = list(plates)
        self._current_plate_collision_radii = list(selected["plate_radii"])
        self._current_rack_half_extents_xy = tuple(selected["rack_half_extents_xy"])

    def _sample_once(self, nominals, rng):
        proposal = super()._sample_once(nominals, rng)
        self.proposals.append(proposal)
        return proposal

    def _contacts_ok(self, model, data):
        self.contact_checks += 1
        return super()._contacts_ok(model, data)


class DishRackPlacementTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

        def variants(kind):
            return list(FIXTURE["variants"][kind])

        self.stack.enter_context(patch.object(assets, "_dishrack_variant_names", variants))
        self.stack.enter_context(patch.object(task, "_dishrack_variant_names", variants))

    def randomizer(self):
        model, data = geometry_model()
        return GeometryDishRack(model, data), model, data

    def assert_recorded_prefix(self, proposals):
        digest = hashlib.sha256(json.dumps(proposals[:200], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, FIXTURE["first_200_proposals_sha256"])

    def test_crowded_scene_reaches_its_first_valid_layout(self):
        randomizer, model, data = self.randomizer()
        state = randomizer.randomize(model, data, seed=FIXTURE["seed"])
        self.assertEqual(len(randomizer.proposals), FIXTURE["accepted_proposal"])
        self.assert_recorded_prefix(randomizer.proposals)
        self.assertEqual(state.object_states, FIXTURE["accepted_state"])
        self.assertEqual(state.scale_states, FIXTURE["selected"]["scales"])
        self.assertEqual(state.metadata["plate_count"], 4)
        self.assertTrue(randomizer._bounds_ok(state.object_states))
        self.assertTrue(randomizer._pairwise_ok(state.object_states))
        self.assertEqual(randomizer.contact_checks, 1)
        self.assertTrue(randomizer._contacts_ok(model, data))

    def test_original_budget_exhausts_without_a_contact_check(self):
        randomizer, model, data = self.randomizer()
        randomizer.max_tries = 200
        with self.assertRaisesRegex(RandomizationSamplingError, "after 200 tries"):
            randomizer.randomize(model, data, seed=FIXTURE["seed"])
        self.assertEqual(len(randomizer.proposals), 200)
        self.assert_recorded_prefix(randomizer.proposals)
        self.assertEqual(randomizer.contact_checks, 0)

    def test_larger_budget_preserves_the_first_accepted_scene(self):
        states = []
        for budget in (FIXTURE["accepted_proposal"], task.DishRackRandomizer.max_tries):
            randomizer, model, data = self.randomizer()
            randomizer.max_tries = budget
            states.append(randomizer.randomize(model, data, seed=FIXTURE["seed"]).to_dict())
        self.assertEqual(states[0], states[1])

    def test_invalid_geometry_still_exhausts_its_finite_budget(self):
        randomizer, model, data = self.randomizer()
        randomizer.max_tries = 5
        randomizer.rack_table_edge_bounds = (0.60, 0.61, -0.01, 0.01)
        with self.assertRaisesRegex(RandomizationSamplingError, "after 5 tries"):
            randomizer.randomize(model, data, seed=FIXTURE["seed"])
        self.assertEqual(len(randomizer.proposals), 5)
        self.assertEqual(randomizer.contact_checks, 0)


if __name__ == "__main__":
    unittest.main()
