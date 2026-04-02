#!/usr/bin/env python3
"""
Generate all ground truth: NAM amp pairs, effect chains, and passthrough.

For each dry guitar file:
  - Process through each NAM model (amp distortion only)
  - Generate random effect chains (amp + reverb/delay/modulation)
  - Create a passthrough copy (dry = dry, teaches model to preserve clean signals)

Writes directly to S3 as files are generated.

SageMaker channels:
  SM_CHANNEL_IDMT       → /opt/ml/input/data/idmt/
  SM_CHANNEL_NAM_MODELS → /opt/ml/input/data/nam_models/
"""

import io
import json
import os
import random
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import soundfile as sf
import torch

from pedalboard import (
    Pedalboard, Reverb, Delay, Chorus, Compressor,
    LowpassFilter,
)

from nam.models.linear import Linear
from nam.models.recurrent import LSTM
from nam.models.wavenet import WaveNet

BUCKET = os.environ.get("DEFX_S3_BUCKET", "YOUR-BUCKET-NAME")
S3_PREFIX = "ground_truth"
CHAINS_PER_FILE = 14      # ~16K chain pairs from ~1173 files
NAM_PAIRS_PER_FILE = 7    # ~8K NAM-only pairs (randomly sampled from available models)
SEED = 42

# --- NAM model loading ---

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


# --- Random effect builders ---

def random_reverb():
    room_size = random.uniform(0.15, 0.85)
    damping = random.uniform(0.3, 0.8)
    wet = random.uniform(0.1, 0.45)
    return Reverb(room_size=room_size, damping=damping,
                  wet_level=wet, dry_level=1.0 - wet)


def random_delay():
    delay_s = random.choice([0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5])
    feedback = random.uniform(0.1, 0.45)
    mix = random.uniform(0.15, 0.35)
    cutoff = random.uniform(2500, 5000)
    return [Delay(delay_seconds=delay_s, feedback=feedback, mix=mix),
            LowpassFilter(cutoff_frequency_hz=cutoff)]


def random_chorus():
    return Chorus(rate_hz=random.uniform(0.5, 2.5),
                  depth=random.uniform(0.1, 0.4),
                  mix=random.uniform(0.2, 0.5),
                  centre_delay_ms=7.0)


def random_compressor():
    return Compressor(threshold_db=random.uniform(-25, -10),
                      ratio=random.choice([2, 3, 4, 6]),
                      attack_ms=random.uniform(5, 30),
                      release_ms=random.uniform(50, 200))


# Chain templates: (builder_fn, weight)
# Higher weight = more likely. ~90% will include reverb.
CHAIN_TEMPLATES = [
    (lambda: ("amp_only", [], []), 1),
    (lambda: ("amp_reverb", [], [random_reverb()]), 10),
    (lambda: ("amp_delay_reverb", [], random_delay() + [random_reverb()]), 6),
    (lambda: ("amp_chorus_reverb", [], [random_chorus(), random_reverb()]), 5),
    (lambda: ("comp_amp_reverb", [random_compressor()], [random_reverb()]), 4),
    (lambda: ("amp_slapback_room", [],
             [Delay(delay_seconds=random.uniform(0.08, 0.15),
                    feedback=random.uniform(0.1, 0.2),
                    mix=random.uniform(0.15, 0.3)),
              Reverb(room_size=random.uniform(0.1, 0.3), damping=0.7,
                     wet_level=random.uniform(0.1, 0.2), dry_level=0.85)]), 4),
    (lambda: ("reverb_only", [], [random_reverb()]), 8),
    (lambda: ("delay_only", [], random_delay()), 3),
    (lambda: ("chorus_reverb", [], [random_chorus(), random_reverb()]), 4),
    (lambda: ("comp_amp_delay_reverb", [random_compressor()],
             random_delay() + [random_reverb()]), 5),
]

NO_AMP_CHAINS = {"reverb_only", "delay_only", "chorus_reverb"}


def pick_chain_template():
    templates, weights = zip(*CHAIN_TEMPLATES)
    return random.choices(templates, weights=weights, k=1)[0]


def apply_chain(audio, sr, nam_model, pre_fx, post_fx, skip_amp):
    x = audio.copy()
    if pre_fx:
        x = Pedalboard(pre_fx)(x[np.newaxis, :], sr)[0]
    if not skip_amp and nam_model is not None:
        x = process_with_nam(nam_model, x)
    if post_fx:
        x = Pedalboard(post_fx)(x[np.newaxis, :], sr)[0]
    return x


# --- S3 helpers ---

def find_wav_files(root: str) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*.wav") if "annotation" not in str(p))


def make_unique_name(wav_path: Path, idmt_root: Path) -> str:
    rel = wav_path.relative_to(idmt_root)
    parts = [p for p in rel.parts if p != "audio"]
    return "_".join(parts).replace(" ", "_").replace(".wav", "")


def upload_wav_to_s3(s3, audio, sr, key):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())


def s3_key_exists(s3, key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except:
        return False


# --- Main ---

def main():
    idmt_dir = Path(os.environ.get("SM_CHANNEL_IDMT", "IDMT-SMT-GUITAR_V2"))
    nam_dir = Path(os.environ.get("SM_CHANNEL_NAM_MODELS", "models/nam"))

    s3 = boto3.client("s3")
    random.seed(SEED)

    wav_files = find_wav_files(idmt_dir)
    nam_files = sorted(nam_dir.glob("*.nam"))

    print(f"IDMT WAVs: {len(wav_files)}")
    print(f"NAM models: {len(nam_files)}")
    print(f"Chains per file: {CHAINS_PER_FILE}")

    # Load NAM models
    models = {}
    for nam_path in nam_files:
        tag = nam_path.stem
        print(f"  Loading NAM: {tag}")
        models[tag] = load_nam_model(str(nam_path))
    nam_tags = list(models.keys())

    total_nam = len(wav_files) * NAM_PAIRS_PER_FILE
    total_chains = len(wav_files) * CHAINS_PER_FILE
    total_pass = len(wav_files)
    print(f"Generating: ~{total_nam} NAM-only + ~{total_chains} chains + {total_pass} passthrough\n")

    t0 = time.time()
    count = 0
    skipped = 0
    errors = 0

    for wav_path in wav_files:
        unique_name = make_unique_name(wav_path, idmt_dir)

        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            print(f"  Error reading {wav_path}: {e}")
            errors += 1
            continue

        mono = audio[:, 0] if audio.ndim > 1 else audio

        # --- Upload dry copy ---
        dry_key = f"{S3_PREFIX}/dry/{unique_name}.wav"
        if not s3_key_exists(s3, dry_key):
            upload_wav_to_s3(s3, mono, sr, dry_key)

        # --- Passthrough pair (dry = dry) ---
        pass_key = f"{S3_PREFIX}/wet/{unique_name}_clean_wet.wav"
        if not s3_key_exists(s3, pass_key):
            s3.copy_object(Bucket=BUCKET,
                           CopySource={"Bucket": BUCKET, "Key": dry_key},
                           Key=pass_key)

        # --- NAM-only pairs (randomly sampled amp models) ---
        sampled_tags = random.sample(nam_tags, min(NAM_PAIRS_PER_FILE, len(nam_tags)))
        for tag in sampled_tags:
            model = models[tag]
            wet_key = f"{S3_PREFIX}/wet/{unique_name}_{tag}_wet.wav"
            if s3_key_exists(s3, wet_key):
                skipped += 1
                count += 1
                continue
            try:
                wet = process_with_nam(model, mono)
                upload_wav_to_s3(s3, wet, sr, wet_key)
                count += 1
            except Exception as e:
                print(f"  Error {wav_path.name} + {tag}: {e}")
                errors += 1

        # --- Random effect chains ---
        for chain_idx in range(CHAINS_PER_FILE):
            template = pick_chain_template()
            chain_name, pre_fx, post_fx = template()
            skip_amp = chain_name in NO_AMP_CHAINS

            nam_tag = random.choice(nam_tags)
            nam_model = models[nam_tag]

            suffix = f"{nam_tag}_{chain_name}_{chain_idx}" if not skip_amp else f"{chain_name}_{chain_idx}"
            wet_key = f"{S3_PREFIX}/wet/{unique_name}_{suffix}_wet.wav"

            if s3_key_exists(s3, wet_key):
                skipped += 1
                count += 1
                continue

            try:
                wet = apply_chain(mono, sr, nam_model, pre_fx, post_fx, skip_amp)
                upload_wav_to_s3(s3, wet, sr, wet_key)
                count += 1
            except Exception as e:
                print(f"  Error {wav_path.name} + {chain_name}: {e}")
                errors += 1

        # Progress
        total_done = count + skipped
        if total_done % 200 == 0 and total_done > 0:
            elapsed = time.time() - t0
            rate = (count - skipped) / max(elapsed, 1)
            print(f"  [{total_done}] {rate:.1f} new/s, skipped {skipped}, errors {errors}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}min")
    print(f"  NAM + chain pairs: {count}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
