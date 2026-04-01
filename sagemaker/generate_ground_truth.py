#!/usr/bin/env python3
"""
SageMaker processing script: generate dry/wet pairs from IDMT WAVs + NAM models.

Writes results directly to S3 as they're generated so progress
is preserved even if the job is interrupted.

SageMaker channels:
  SM_CHANNEL_IDMT       → /opt/ml/input/data/idmt/
  SM_CHANNEL_NAM_MODELS → /opt/ml/input/data/nam_models/
"""

import io
import json
import os
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import soundfile as sf
import torch

from nam.models.linear import Linear
from nam.models.recurrent import LSTM
from nam.models.wavenet import WaveNet

BUCKET = os.environ.get("DEFX_S3_BUCKET", "defx-629711664886")
S3_PREFIX = "ground_truth"

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


def load_nam_model(nam_path: str) -> torch.nn.Module:
    with open(nam_path, "r") as f:
        config = json.load(f)
    arch = config["architecture"]
    sample_rate = config.get("sample_rate", None)
    model = _BUILDERS[arch](config["config"], sample_rate)
    model.import_weights(torch.Tensor(config["weights"]))
    model.eval()
    return model


def process_with_nam(model, audio: np.ndarray) -> np.ndarray:
    squeeze = audio.ndim == 1
    if squeeze:
        audio = audio[np.newaxis, :]
    results = []
    with torch.no_grad():
        for ch in range(audio.shape[0]):
            x = torch.tensor(audio[ch], dtype=torch.float32).unsqueeze(0)
            y = model(x, pad_start=True)
            results.append(y.squeeze(0).numpy())
    out = np.stack(results, axis=0)
    return out[0] if squeeze else out


def find_wav_files(root: str) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*.wav") if "annotation" not in str(p))


def make_unique_name(wav_path: Path, idmt_root: Path) -> str:
    rel = wav_path.relative_to(idmt_root)
    parts = [p for p in rel.parts if p != "audio"]
    return "_".join(parts).replace(" ", "_").replace(".wav", "")


def upload_wav_to_s3(s3_client, audio: np.ndarray, sr: int, s3_key: str):
    """Write a WAV to an in-memory buffer and upload to S3."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    s3_client.put_object(Bucket=BUCKET, Key=s3_key, Body=buf.getvalue())


def s3_key_exists(s3_client, key: str) -> bool:
    """Check if an S3 key already exists (for resumability)."""
    try:
        s3_client.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def main():
    idmt_dir = Path(os.environ.get("SM_CHANNEL_IDMT", "IDMT-SMT-GUITAR_V2"))
    nam_dir = Path(os.environ.get("SM_CHANNEL_NAM_MODELS", "models/nam"))

    s3 = boto3.client("s3")

    wav_files = find_wav_files(idmt_dir)
    nam_files = sorted(nam_dir.glob("*.nam"))

    print(f"IDMT WAVs: {len(wav_files)}")
    print(f"NAM models: {len(nam_files)}")
    print(f"Total pairs to generate: {len(wav_files) * len(nam_files)}")
    print(f"Output: s3://{BUCKET}/{S3_PREFIX}/\n")

    # Load all NAM models
    models = {}
    for nam_path in nam_files:
        tag = nam_path.stem
        print(f"  Loading NAM: {tag}")
        models[tag] = load_nam_model(str(nam_path))

    t0 = time.time()
    count = 0
    skipped = 0
    errors = 0
    total = len(wav_files) * len(models)

    for wav_path in wav_files:
        unique_name = make_unique_name(wav_path, idmt_dir)

        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            print(f"  Error reading {wav_path}: {e}")
            errors += 1
            continue

        # Ensure mono
        if audio.ndim > 1:
            mono = audio[:, 0]
        else:
            mono = audio

        # Upload dry copy (once per WAV)
        dry_key = f"{S3_PREFIX}/dry/{unique_name}.wav"
        if not s3_key_exists(s3, dry_key):
            upload_wav_to_s3(s3, mono, sr, dry_key)

        # Also create a "passthrough" wet pair (dry=dry) so the model
        # learns to leave clean signals unchanged
        passthrough_key = f"{S3_PREFIX}/wet/{unique_name}_clean_wet.wav"
        if not s3_key_exists(s3, passthrough_key):
            upload_wav_to_s3(s3, mono, sr, passthrough_key)

        # Process through each NAM model
        for tag, model in models.items():
            wet_key = f"{S3_PREFIX}/wet/{unique_name}_{tag}_wet.wav"

            if s3_key_exists(s3, wet_key):
                skipped += 1
                count += 1
                continue

            try:
                wet = process_with_nam(model, mono)
                upload_wav_to_s3(s3, wet, sr, wet_key)
                count += 1

                if count % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (count - skipped) / max(elapsed, 1)
                    remaining = (total - count) / max(rate, 0.01) / 60
                    print(f"  [{count}/{total}] {rate:.1f} new/s, "
                          f"skipped {skipped}, ETA {remaining:.0f}min")

            except Exception as e:
                print(f"  Error {wav_path.name} + {tag}: {e}")
                errors += 1

    elapsed = time.time() - t0
    print(f"\nDone. {count} pairs ({skipped} skipped) in {elapsed/60:.1f}min, {errors} errors")


if __name__ == "__main__":
    main()
