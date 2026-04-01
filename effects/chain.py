"""
Effect chain: run multiple effects in series.

Each effect in the chain processes the output of the previous one,
just like a real pedalboard signal path.
"""

import numpy as np


class EffectChain:
    """
    A series of effects applied sequentially.

    Each effect must be callable as: wet = effect(audio, sample_rate)
    """

    def __init__(self, effects: list[tuple] | None = None):
        # List of (effect, name) tuples
        self._effects: list[tuple] = effects or []

    def add(self, effect, name: str = ""):
        self._effects.append((effect, name))

    def __call__(self, audio: np.ndarray, sample_rate: float) -> np.ndarray:
        for effect, _ in self._effects:
            audio = effect(audio, sample_rate)
        return audio

    @property
    def names(self) -> list[str]:
        return [name for _, name in self._effects]

    def __repr__(self) -> str:
        return " -> ".join(self.names) if self._effects else "(empty chain)"

    def __len__(self) -> int:
        return len(self._effects)
