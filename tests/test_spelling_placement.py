"""Conservative placement counterexamples retained from the shipped objective."""

import string
import unittest
from copy import deepcopy
from dataclasses import replace
from itertools import permutations

import numpy as np

from abc_sim.spelling_placement import FreePlacementSpellingCfg, score_free_placement


def spelling_frame(word="ABC", row="ABC"):
    letters = list(string.ascii_uppercase)
    initial = [
        [0.4 + 0.1 * (index // 9), 0.4 - 0.085 * (index % 9), 0.02]
        for index in range(26)
    ]
    positions = deepcopy(initial)
    for index, letter in enumerate(row):
        if letter != "_":
            positions[letters.index(letter)] = [0.0, 0.1 - 0.04 * index, 0.02]
    return {
        "word": word,
        "letters": letters,
        "positions": positions,
        "initial_positions": initial,
        "face_up": [True] * 26,
        "supported": [True] * 26,
    }


def score(frame):
    return score_free_placement(FreePlacementSpellingCfg(), **frame)


class SpellingPlacementTests(unittest.TestCase):
    def test_complete_rows_and_every_permutation(self):
        for word in ("AB", "ABC", "ABCD"):
            for permutation in permutations(word):
                result = score(spelling_frame(word, "".join(permutation)))
                expected = sorted(
                    letter
                    for letter, target in zip(permutation, word, strict=True)
                    if letter == target
                )
                self.assertEqual(result["credited_letters"], expected)
                self.assertEqual(result["success"], tuple(word) == permutation)
            self.assertEqual(score(spelling_frame(word, word))["reward"], 1)

    def test_wrong_occupants_and_missing_endpoints(self):
        for word, row, credited, value in (
            ("AB", "XB", ["B"], 0.5),
            ("AB", "_B", [], 0),
            ("AB", "A_", [], 0),
            ("ABC", "XBC", ["B", "C"], 2 / 3),
            ("ABC", "AXC", ["A", "C"], 2 / 3),
            ("ABC", "ABX", ["A", "B"], 2 / 3),
            ("ABC", "_BC", [], 0),
            ("ABC", "AB_", [], 0),
            ("ABC", "A_C", ["A", "C"], 2 / 3),
        ):
            with self.subTest(word=word, row=row):
                result = score(spelling_frame(word, row))
                self.assertEqual(result["credited_letters"], credited)
                self.assertEqual(result["reward"], value)
                self.assertFalse(result["success"])
        for row in ("_BC", "AB_"):
            self.assertEqual(score(spelling_frame(row=row))["interpretations"], 2)

    def test_extra_held_or_colliding_occupants_prevent_completion(self):
        for row in ("XABC", "ABCX", "AXBC"):
            self.assertLess(score(spelling_frame(row=row))["reward"], 1)
        frame = spelling_frame()
        index = frame["letters"].index("X")
        frame["positions"][index] = [0.0, 0.08, 0.2]
        frame["supported"][index] = False
        self.assertFalse(score(frame)["success"])
        frame["positions"][index] = frame["positions"][1].copy()
        self.assertEqual(score(frame)["reward"], 0)

    def test_untouched_scattered_and_competing_rows_do_not_choose_targets(self):
        frame = spelling_frame()
        frame["positions"] = deepcopy(frame["initial_positions"])
        self.assertEqual(score(frame)["reward"], 0)
        frame = spelling_frame()
        for index in range(3):
            frame["positions"][index][0] = -0.3 * index
        self.assertEqual(score(frame)["reward"], 0)
        frame = spelling_frame()
        for index, letter in enumerate("XYZ"):
            frame["positions"][frame["letters"].index(letter)] = [
                -0.3,
                0.1 - index * 0.04,
                0.02,
            ]
        self.assertEqual(len(score(frame)["rows"]), 2)
        self.assertEqual(score(frame)["reward"], 0)

    def test_ineligible_target_still_occupies_its_slot(self):
        for flag in ("face_up", "supported"):
            frame = spelling_frame()
            frame[flag][1] = False
            result = score(frame)
            self.assertEqual(result["credited_letters"], ["A", "C"])
            self.assertEqual(result["reward"], 2 / 3)

    def test_translation_and_inventory_order_do_not_change_result(self):
        frame = spelling_frame(row="XBC")
        original = score(frame)
        for key in ("positions", "initial_positions"):
            frame[key] = (np.asarray(frame[key]) + [0.13, -0.22, 0.1]).tolist()
        for key in (
            "letters",
            "positions",
            "initial_positions",
            "face_up",
            "supported",
        ):
            frame[key].reverse()
        self.assertEqual(score(frame), original)

    def test_nonfinite_or_incomplete_evidence_fails(self):
        for key in ("positions", "initial_positions"):
            frame = spelling_frame()
            frame[key][0][0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                score(frame)
        frame = spelling_frame()
        frame["supported"][0] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            score(frame)

    def test_configuration_is_finite_versioned_and_ordered(self):
        cfg = FreePlacementSpellingCfg()
        for fields in (
            {"revision": "v2"},
            {"gap_min_m": 0.1},
            {"row_width_m": float("nan")},
            {"minimum_weight_fraction": 1.1},
            {"max_angular_speed": 0},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(cfg, **fields)


if __name__ == "__main__":
    unittest.main()
