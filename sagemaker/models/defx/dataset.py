"""
Dataset for DeFX training: loads aligned dry/wet WAV pairs,
returns random fixed-length chunks.

Supports both pre-loading (fast, for small datasets) and
lazy-loading (memory-efficient, for large datasets).
"""

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class DryWetDataset(Dataset):
    """
    Loads paired dry/wet audio files and serves random chunks.

    Args:
        pairs: list of (dry_path, wet_path) tuples
        chunk_samples: length of each training chunk in samples
        sample_rate: expected sample rate
        augment: apply random gain augmentation
        stereo: output stereo (2ch) or mono
        preload: if True, load all audio into RAM (fast but memory-heavy)
                 if False, read from disk on each access (slower but scalable)
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        chunk_samples: int = 65536,
        sample_rate: int = 44100,
        augment: bool = True,
        stereo: bool = True,
        preload: bool = False,
    ):
        self.pairs = pairs
        self.chunk_samples = chunk_samples
        self.sample_rate = sample_rate
        self.augment = augment
        self.stereo = stereo
        self.preload = preload

        if preload:
            self.data = []
            for dry_path, wet_path in pairs:
                dry, wet = self._load_pair(dry_path, wet_path)
                self.data.append((dry, wet))
            self._total_chunks = sum(
                max(1, (d.shape[-1] - self.chunk_samples) // (self.chunk_samples // 2))
                for d, _ in self.data
            )
        else:
            # Scan file lengths without loading audio
            self._lengths = []
            for dry_path, _ in pairs:
                info = sf.info(dry_path)
                self._lengths.append(info.frames)
            self._total_chunks = sum(
                max(1, (l - self.chunk_samples) // (self.chunk_samples // 2))
                for l in self._lengths
            )

    def _load_pair(self, dry_path: str, wet_path: str):
        """Load and format a dry/wet pair."""
        dry, _ = sf.read(dry_path, dtype="float32", always_2d=True)
        wet, _ = sf.read(wet_path, dtype="float32", always_2d=True)

        if self.stereo:
            if dry.shape[1] == 1:
                dry = np.repeat(dry, 2, axis=1)
            if wet.shape[1] == 1:
                wet = np.repeat(wet, 2, axis=1)
            dry = dry[:, :2].T
            wet = wet[:, :2].T
        else:
            dry = dry[:, 0]
            wet = wet[:, 0]

        min_len = min(dry.shape[-1], wet.shape[-1])
        dry = dry[..., :min_len]
        wet = wet[..., :min_len]
        return dry, wet

    def __len__(self) -> int:
        return max(self._total_chunks * 2, 64)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Pick a random file pair
        pair_idx = random.randint(0, len(self.pairs) - 1)

        if self.preload:
            dry, wet = self.data[pair_idx]
        else:
            dry_path, wet_path = self.pairs[pair_idx]
            dry, wet = self._load_pair(dry_path, wet_path)

        length = dry.shape[-1]

        # Random offset
        max_start = length - self.chunk_samples
        if max_start <= 0:
            pad_width = self.chunk_samples - length
            if self.stereo:
                dry = np.pad(dry, ((0, 0), (0, pad_width)))
                wet = np.pad(wet, ((0, 0), (0, pad_width)))
            else:
                dry = np.pad(dry, (0, pad_width))
                wet = np.pad(wet, (0, pad_width))
            start = 0
        else:
            start = random.randint(0, max_start)

        dry_chunk = dry[..., start : start + self.chunk_samples].copy()
        wet_chunk = wet[..., start : start + self.chunk_samples].copy()

        if self.augment:
            gain = random.uniform(0.5, 1.5)
            dry_chunk = dry_chunk * gain
            wet_chunk = wet_chunk * gain

        return (
            torch.from_numpy(wet_chunk),   # input (wet)
            torch.from_numpy(dry_chunk),   # target (dry)
        )
