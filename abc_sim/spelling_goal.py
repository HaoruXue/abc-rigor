"""Reproducible goals for scenes with one block of each uppercase letter."""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpellingGoal:
    """One exact sequence, with a seed and versioned sampling rule."""

    sequence: str
    seed: int
    generator_revision: str = "unique-uppercase-v1"

    def __post_init__(self) -> None:
        if self.generator_revision != "unique-uppercase-v1":
            raise ValueError("Unsupported spelling goal generator.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("A spelling goal requires a nonnegative integer seed.")
        if (
            type(self.sequence) is not str
            or len(self.sequence) not in (2, 3, 4)
            or any(letter not in string.ascii_uppercase for letter in self.sequence)
            or len(set(self.sequence)) != len(self.sequence)
        ):
            raise ValueError("A spelling goal requires 2, 3, or 4 distinct uppercase letters.")

    @property
    def prompt(self) -> str:
        return f"spell the letter sequence {self.sequence.lower()}"

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "spelling_goal": {**asdict(self), "digest": self.digest},
            "prompt": self.prompt,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> SpellingGoal:
        record = dict(metadata["spelling_goal"])
        digest = record.pop("digest")
        goal = cls(**record)
        if digest != goal.digest or metadata["prompt"] != goal.prompt:
            raise ValueError("Spelling goal content, digest and prompt must agree.")
        return goal

    @classmethod
    def sample(cls, length: int, seed: int) -> SpellingGoal:
        if type(length) is not int or length not in (2, 3, 4):
            raise ValueError("Spelling sequence length must be 2, 3, or 4.")
        if type(seed) is not int or seed < 0:
            raise ValueError("Random spelling requires an explicit nonnegative integer seed.")
        # The goal stream is independent of collision-rejection draws for the scene.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5350454C]))
        sequence = "".join(rng.choice(list(string.ascii_uppercase), size=length, replace=False))
        return cls(sequence=sequence, seed=seed)
