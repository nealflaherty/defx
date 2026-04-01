"""
Minimal NAM model loader for offline/batch inference.

The `neural-amp-modeler` pip package eagerly imports tkinter (for its
training GUI), which breaks headless usage. This module imports only
the model classes we need — pure PyTorch, no GUI dependencies.
"""

import json
from pathlib import Path

import numpy as np
import torch

# Import the model classes directly — these are pure PyTorch
# and don't trigger the tkinter import chain.
from nam.models.linear import Linear
from nam.models.recurrent import LSTM
from nam.models.wavenet import WaveNet


_BUILDERS = {
    "Linear": lambda cfg, sr: Linear(sample_rate=sr, **cfg),
    "LSTM": lambda cfg, sr: LSTM(sample_rate=sr, **cfg),
    "WaveNet": lambda cfg, sr: WaveNet(
        layers_configs=cfg["layers"],
        head_config=cfg["head"],
        head_scale=cfg["head_scale"],
        sample_rate=sr,
    ),
}


def load_nam_model(nam_path: str | Path) -> torch.nn.Module:
    """
    Load a .nam file and return a ready-to-use PyTorch model.

    Args:
        nam_path: Path to a .nam JSON file.

    Returns:
        A PyTorch nn.Module that accepts (batch, samples) tensors.
    """
    nam_path = Path(nam_path)
    if not nam_path.exists():
        raise FileNotFoundError(f"NAM model not found: {nam_path}")

    with open(nam_path, "r") as f:
        config = json.load(f)

    arch = config["architecture"]
    if arch not in _BUILDERS:
        raise ValueError(
            f"Unknown NAM architecture '{arch}'. "
            f"Supported: {list(_BUILDERS.keys())}"
        )

    sample_rate = config.get("sample_rate", None)
    model = _BUILDERS[arch](config["config"], sample_rate)
    model.import_weights(torch.Tensor(config["weights"]))
    model.eval()
    return model


class NAMEffect:
    """
    Wraps a NAM model to match the calling convention used by our pipeline:
        wet = effect(audio_np_array, sample_rate)

    Handles numpy<->torch conversion and batching.
    """

    def __init__(self, nam_path: str | Path):
        self.model = load_nam_model(nam_path)
        self.nam_path = Path(nam_path)

    def __call__(self, audio: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Process audio through the NAM model.

        Args:
            audio: numpy array, shape (channels, samples) or (samples,)
            sample_rate: sample rate in Hz

        Returns:
            Processed audio as numpy array, same shape as input.
        """
        # Handle mono (samples,) or multi-channel (channels, samples)
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[np.newaxis, :]  # (1, samples)

        results = []
        with torch.no_grad():
            for ch in range(audio.shape[0]):
                x = torch.tensor(audio[ch], dtype=torch.float32).unsqueeze(0)
                y = self.model(x, pad_start=True)
                results.append(y.squeeze(0).numpy())

        out = np.stack(results, axis=0)
        if squeeze:
            out = out[0]
        return out

    def __repr__(self) -> str:
        return f"NAMEffect({self.nam_path.name})"
