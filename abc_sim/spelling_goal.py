"""Seeded goals for scenes with one block of each uppercase letter."""

from __future__ import annotations

import string
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpellingGoal:
    """A realized sequence of two, three, or four distinct letter blocks."""

    sequence: str
    seed: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("A spelling goal requires a nonnegative integer seed.")
        if (
            not isinstance(self.sequence, str)
            or len(self.sequence) not in (2, 3, 4)
            or any(letter not in string.ascii_uppercase for letter in self.sequence)
            or len(set(self.sequence)) != len(self.sequence)
        ):
            raise ValueError(
                "A spelling goal requires 2, 3, or 4 distinct uppercase letters."
            )

    @property
    def prompt(self) -> str:
        return f"spell the letter sequence {self.sequence.lower()}"

    def to_metadata(self) -> dict[str, Any]:
        return {"spelling_goal": asdict(self), "prompt": self.prompt}

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> SpellingGoal:
        goal = cls(**metadata["spelling_goal"])
        if metadata["prompt"] != goal.prompt:
            raise ValueError("Spelling goal and prompt must agree.")
        return goal

    @classmethod
    def sample(cls, length: int, seed: int) -> SpellingGoal:
        if type(length) is not int or length not in (2, 3, 4):
            raise ValueError("Spelling sequence length must be 2, 3, or 4.")
        if type(seed) is not int or seed < 0:
            raise ValueError(
                "Random spelling requires an explicit nonnegative integer seed."
            )
        # A separate stream keeps the goal independent of scene-placement retries.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5350454C]))
        sequence = "".join(
            rng.choice(list(string.ascii_uppercase), size=length, replace=False)
        )
        return cls(sequence=sequence, seed=seed)
