#!/usr/bin/env python3
"""
Run DeFX inference: remove effects from a guitar recording.

Usage:
    python inference.py --input wet_guitar.wav --output clean_guitar.wav
    python inference.py --input wet_guitar.wav --checkpoint models/defx/checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from models.defx.demucs_defx import DemucsDefx


def load_model(checkpoint_path: str, device: str = "cpu") -> DemucsDefx:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    unfreeze_dec = ckpt.get("unfreeze_decoder_layers", 0)
    unfreeze_enc = ckpt.get("unfreeze_encoder_layers", 0)
    model = DemucsDefx(
        freeze_encoder=True,
        unfreeze_decoder_layers=unfreeze_dec,
        unfreeze_encoder_layers=unfreeze_enc,
    ).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    if "backbone_state_dict" in ckpt and ckpt["backbone_state_dict"]:
        model.backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    print(f"Loaded: epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
    model.eval()
    return model


def inference(model, wet_path: str, output_path: str, device: str = "cpu"):
    wet_np, sr = sf.read(wet_path, dtype="float32", always_2d=True)
    print(f"Input: {wet_path} ({sr}Hz, {wet_np.shape[0]} samples, {wet_np.shape[1]}ch)")

    if wet_np.shape[1] == 1:
        wet_np = np.repeat(wet_np, 2, axis=1)
    wet_np = wet_np[:, :2].T

    with torch.no_grad():
        wet_t = torch.from_numpy(wet_np).unsqueeze(0).to(device)
        pred_t = model(wet_t)
        pred_np = pred_t.squeeze(0).cpu().numpy()

    sf.write(output_path, pred_np.T, sr)
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DeFX: remove effects from guitar audio")
    parser.add_argument("--input", type=str, required=True, help="Wet/effected audio file")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--checkpoint", type=str, default="models/defx/checkpoints/best.pt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"{stem}_clean.wav"

    model = load_model(args.checkpoint, device)
    inference(model, args.input, args.output, device)


if __name__ == "__main__":
    main()
