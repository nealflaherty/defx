#!/usr/bin/env python3
"""
Evaluate a trained DeFX checkpoint and produce metrics + figures for the paper.

Usage:
    python evaluate.py --dry-dir data/dry --wet-dir data/wet
    python evaluate.py --dry-dir data/dry --wet-dir data/wet --checkpoint models/defx/checkpoints/demucs_defx_best.pt --output-dir docs/figures

Produces:
    - Objective metrics table (CSV + printed)
    - Per-effect-type SI-SDR bar chart (PDF)
    - Per-distortion-level SI-SDR bar chart (PDF)
    - Spectrogram comparison figures (PDF)
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import auraloss.freq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.defx.demucs_defx import DemucsDefx


# ---------------------------------------------------------------------------
# Model loading (mirrors inference.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> DemucsDefx:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = DemucsDefx(
        freeze_encoder=True,
        unfreeze_decoder_layers=ckpt.get("unfreeze_decoder_layers", 0),
        unfreeze_encoder_layers=ckpt.get("unfreeze_encoder_layers", 0),
    ).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    if "backbone_state_dict" in ckpt and ckpt["backbone_state_dict"]:
        model.backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB."""
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)
    min_len = min(len(ref), len(est))
    ref, est = ref[:min_len], est[:min_len]
    dot = np.dot(ref, est)
    s_target = dot * ref / (np.dot(ref, ref) + 1e-8)
    e_noise = est - s_target
    return 10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8))


def compute_l1(estimate: np.ndarray, reference: np.ndarray) -> float:
    min_len = min(len(estimate), len(reference))
    return np.mean(np.abs(estimate[:min_len] - reference[:min_len]))


def compute_mrstft(estimate: np.ndarray, reference: np.ndarray, mrstft_fn) -> float:
    min_len = min(len(estimate), len(reference))
    est_t = torch.from_numpy(estimate[:min_len]).float().unsqueeze(0).unsqueeze(0)
    ref_t = torch.from_numpy(reference[:min_len]).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return mrstft_fn(est_t, ref_t).item()


def compute_mel_cepstral_distortion(estimate: np.ndarray, reference: np.ndarray,
                                     sr: int = 44100, n_mels: int = 40,
                                     n_fft: int = 2048, hop: int = 512) -> float:
    """Mel cepstral distortion (MCD) in dB."""
    from scipy.fftpack import dct

    def mel_cepstrum(audio):
        # Simple mel spectrogram via numpy
        n_frames = 1 + (len(audio) - n_fft) // hop
        if n_frames < 1:
            return None
        frames = np.stack([audio[i * hop : i * hop + n_fft] for i in range(n_frames)])
        windowed = frames * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2

        # Mel filterbank
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        mel_lo = 2595 * np.log10(1 + freqs[0] / 700)
        mel_hi = 2595 * np.log10(1 + freqs[-1] / 700)
        mel_points = np.linspace(mel_lo, mel_hi, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(n_mels):
            for k in range(bins[m], bins[m + 1]):
                if bins[m + 1] != bins[m]:
                    fbank[m, k] = (k - bins[m]) / (bins[m + 1] - bins[m])
            for k in range(bins[m + 1], bins[m + 2]):
                if bins[m + 2] != bins[m + 1]:
                    fbank[m, k] = (bins[m + 2] - k) / (bins[m + 2] - bins[m + 1])

        mel_spec = np.dot(spec, fbank.T)
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel = np.log(mel_spec)
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, 1:14]
        return mfcc

    min_len = min(len(estimate), len(reference))
    mc_est = mel_cepstrum(estimate[:min_len])
    mc_ref = mel_cepstrum(reference[:min_len])
    if mc_est is None or mc_ref is None:
        return float("nan")
    n = min(len(mc_est), len(mc_ref))
    diff = mc_est[:n] - mc_ref[:n]
    return (10.0 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Filename parsing — extract effect type and distortion level from wet names
# ---------------------------------------------------------------------------

CHAIN_NAMES = [
    "amp_only", "amp_reverb", "amp_delay_reverb", "amp_chorus_reverb",
    "comp_amp_reverb", "amp_slapback_room", "reverb_only", "delay_only",
    "chorus_reverb", "comp_amp_delay_reverb",
]

# Volume ranges for distortion level grouping
GAIN_LEVELS = {
    "clean":   (0, 3.5),
    "breakup": (3.5, 6.0),
    "crunch":  (6.0, 8.5),
    "cranked":  (8.5, 11.0),
}


def classify_effect(wet_name: str) -> str:
    """Extract effect chain type from wet filename."""
    if wet_name.endswith("_clean_wet"):
        return "passthrough"
    for chain in sorted(CHAIN_NAMES, key=len, reverse=True):
        if f"_{chain}_" in wet_name:
            return chain
    # NAM-only pair (has blackpanel tag but no chain name)
    if "blackpanel_" in wet_name:
        return "nam_only"
    return "unknown"


def extract_volume(wet_name: str) -> float | None:
    """Extract amp volume from NAM model tag in filename."""
    m = re.search(r"_v(\d+\.?\d*)_t", wet_name)
    return float(m.group(1)) if m else None


def classify_gain(volume: float | None) -> str:
    if volume is None:
        return "unknown"
    for label, (lo, hi) in GAIN_LEVELS.items():
        if lo <= volume < hi:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Pair finding (mirrors train script)
# ---------------------------------------------------------------------------

def find_pairs(dry_dir: str, wet_dir: str) -> list[tuple[str, str]]:
    dry_dir, wet_dir = Path(dry_dir), Path(wet_dir)
    dry_files = sorted(dry_dir.glob("*.wav"))
    wet_files = {p.name: p for p in sorted(wet_dir.glob("*.wav"))}
    pairs = []
    for dry_path in dry_files:
        stem = dry_path.stem
        matched = [wp for wn, wp in wet_files.items()
                   if wn.startswith(stem) and wn.endswith("_wet.wav")]
        for wet_path in matched:
            pairs.append((str(dry_path), str(wet_path)))
    return pairs


def split_pairs(pairs: list[tuple[str, str]], val_fraction: float = 0.1):
    """Split pairs into train/val by dry file to prevent leakage.
    Mirrors the logic in sagemaker/train_demucs_defx.py exactly."""
    import hashlib
    by_dry = {}
    for dry_path, wet_path in pairs:
        by_dry.setdefault(dry_path, []).append((dry_path, wet_path))

    dry_files = sorted(by_dry.keys())
    train_pairs, val_pairs = [], []
    for dry_path in dry_files:
        h = int(hashlib.md5(dry_path.encode()).hexdigest(), 16) % 100
        if h < val_fraction * 100:
            val_pairs.extend(by_dry[dry_path])
        else:
            train_pairs.extend(by_dry[dry_path])

    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Run model on a single file
# ---------------------------------------------------------------------------

def run_model(model, wet_np: np.ndarray, sr: int, device: str) -> np.ndarray:
    """Run DeFX on mono audio, return mono output."""
    stereo = np.stack([wet_np, wet_np])  # (2, T)
    with torch.no_grad():
        wet_t = torch.from_numpy(stereo).float().unsqueeze(0).to(device)
        pred_t = model(wet_t)
        pred_np = pred_t.squeeze(0).cpu().numpy()
    return pred_np[0]  # left channel (mono)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bar_chart(data: dict[str, list[float]], title: str, ylabel: str,
                   output_path: str):
    """Horizontal bar chart with mean + std error bars."""
    labels = sorted(data.keys())
    means = [np.mean(data[k]) for k in labels]
    stds = [np.std(data[k]) / max(np.sqrt(len(data[k])), 1) for k in labels]
    counts = [len(data[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(6, max(4, len(labels) * 0.45)))
    y = np.arange(len(labels))
    bars = ax.barh(y, means, xerr=stds, capsize=3, color="#4C72B0", edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  (n={c})" for l, c in zip(labels, counts)], fontsize=9)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_spectrogram_comparison(wet_np, restored_np, dry_np, sr, title, output_path,
                                 n_fft=2048, hop=512, max_freq_khz=8.0):
    """Vertically stacked spectrograms: wet, restored, dry."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True, sharey=True)

    max_bin = int(max_freq_khz * 1000 * n_fft / sr)

    for ax, audio, label in zip(axes, [wet_np, restored_np, dry_np],
                                 ["Wet (input)", "DeFX (restored)", "Dry (ground truth)"]):
        min_len = min(len(audio), sr * 5)  # cap at 5 seconds for readability
        audio = audio[:min_len]
        n_frames = 1 + (len(audio) - n_fft) // hop
        if n_frames < 1:
            continue
        frames = np.stack([audio[i * hop : i * hop + n_fft] for i in range(n_frames)])
        windowed = frames * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(windowed, n=n_fft))
        log_spec = np.log1p(spec[:, :max_bin]).T

        ax.imshow(log_spec, aspect="auto", origin="lower",
                  extent=[0, min_len / sr, 0, max_freq_khz])
        ax.set_ylabel("Freq (kHz)")
        ax.set_title(label, fontsize=11)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DeFX and produce paper figures")
    parser.add_argument("--dry-dir", type=str, required=True)
    parser.add_argument("--wet-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str,
                        default="models/defx/checkpoints/demucs_defx_best.pt")
    parser.add_argument("--output-dir", type=str, default="docs/figures")
    parser.add_argument("--max-pairs", type=int, default=500,
                        help="Max pairs to evaluate (0=all)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--spectrogram-examples", type=int, default=4,
                        help="Number of spectrogram comparisons to generate")
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
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)

    # Find pairs — evaluate only on validation split to avoid data leakage
    all_pairs = find_pairs(args.dry_dir, args.wet_dir)
    if not all_pairs:
        print("No pairs found!")
        sys.exit(1)

    _, pairs = split_pairs(all_pairs, val_fraction=0.1)
    print(f"Total pairs: {len(all_pairs)}, Validation pairs: {len(pairs)}")

    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pairs), args.max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]

    print(f"Evaluating {len(pairs)} pairs...\n")

    # Setup MRSTFT loss
    mrstft_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_lengths=[512, 1024, 2048],
    )

    # Collect metrics
    all_results = []
    by_effect = defaultdict(list)
    by_gain = defaultdict(list)
    spectrogram_candidates = []

    for i, (dry_path, wet_path) in enumerate(pairs):
        wet_name = Path(wet_path).stem
        effect_type = classify_effect(wet_name)
        volume = extract_volume(wet_name)
        gain_level = classify_gain(volume)

        try:
            dry_np, sr = sf.read(dry_path, dtype="float32")
            wet_np, _ = sf.read(wet_path, dtype="float32")
        except Exception as e:
            print(f"  Error reading pair: {e}")
            continue

        if dry_np.ndim > 1:
            dry_np = dry_np[:, 0]
        if wet_np.ndim > 1:
            wet_np = wet_np[:, 0]

        # Run model
        restored_np = run_model(model, wet_np, sr, device)

        # Compute metrics
        min_len = min(len(dry_np), len(wet_np), len(restored_np))
        dry_np = dry_np[:min_len]
        wet_np = wet_np[:min_len]
        restored_np = restored_np[:min_len]

        sisdr_wet = si_sdr(wet_np, dry_np)
        sisdr_restored = si_sdr(restored_np, dry_np)
        sisdr_improvement = sisdr_restored - sisdr_wet

        l1_wet = compute_l1(wet_np, dry_np)
        l1_restored = compute_l1(restored_np, dry_np)

        mrstft_wet = compute_mrstft(wet_np, dry_np, mrstft_fn)
        mrstft_restored = compute_mrstft(restored_np, dry_np, mrstft_fn)

        mcd_wet = compute_mel_cepstral_distortion(wet_np, dry_np, sr)
        mcd_restored = compute_mel_cepstral_distortion(restored_np, dry_np, sr)

        result = {
            "wet_name": wet_name,
            "effect_type": effect_type,
            "gain_level": gain_level,
            "sisdr_wet": sisdr_wet,
            "sisdr_restored": sisdr_restored,
            "sisdr_improvement": sisdr_improvement,
            "l1_wet": l1_wet,
            "l1_restored": l1_restored,
            "mrstft_wet": mrstft_wet,
            "mrstft_restored": mrstft_restored,
            "mcd_wet": mcd_wet,
            "mcd_restored": mcd_restored,
        }
        all_results.append(result)
        if effect_type == "passthrough":
            # For passthrough, track absolute SI-SDR (not improvement)
            by_effect[effect_type].append(sisdr_restored)
        else:
            by_effect[effect_type].append(sisdr_improvement)
        if gain_level != "unknown":
            by_gain[gain_level].append(sisdr_improvement)

        # Save candidates for spectrogram figures
        if effect_type not in [c[0] for c in spectrogram_candidates]:
            spectrogram_candidates.append((effect_type, wet_np, restored_np, dry_np, sr, wet_name))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(pairs)}]")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OBJECTIVE METRICS SUMMARY")
    print("=" * 70)

    metrics = {
        "SI-SDR (dB)": ("sisdr_wet", "sisdr_restored"),
        "L1": ("l1_wet", "l1_restored"),
        "MRSTFT": ("mrstft_wet", "mrstft_restored"),
        "MCD (dB)": ("mcd_wet", "mcd_restored"),
    }

    print(f"{'Metric':<15} {'Wet (input)':>15} {'Restored':>15} {'Δ':>12}")
    print("-" * 57)
    for name, (wet_key, res_key) in metrics.items():
        wet_vals = [r[wet_key] for r in all_results if not np.isnan(r[wet_key])]
        res_vals = [r[res_key] for r in all_results if not np.isnan(r[res_key])]
        wet_mean = np.mean(wet_vals)
        res_mean = np.mean(res_vals)
        delta = res_mean - wet_mean
        sign = "+" if delta > 0 else ""
        # For L1, MRSTFT, MCD: lower is better, so negative delta is good
        print(f"{name:<15} {wet_mean:>15.4f} {res_mean:>15.4f} {sign}{delta:>11.4f}")

    # Save CSV
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        header = list(all_results[0].keys())
        f.write(",".join(header) + "\n")
        for r in all_results:
            f.write(",".join(str(r[k]) for k in header) + "\n")
    print(f"\nDetailed results: {csv_path}")

    # -----------------------------------------------------------------------
    # Per-effect-type bar chart
    # -----------------------------------------------------------------------
    print("\nPer-effect SI-SDR improvement:")
    for etype in sorted(by_effect.keys()):
        vals = by_effect[etype]
        if etype == "passthrough":
            print(f"  {etype:<25} {np.mean(vals):>7.2f} dB absolute SI-SDR  (n={len(vals)})")
        else:
            print(f"  {etype:<25} {np.mean(vals):>+7.2f} dB  (n={len(vals)})")

    plot_bar_chart(
        by_effect,
        "SI-SDR Improvement by Effect Type",
        "SI-SDR Improvement (dB)",
        str(output_dir / "per_effect_sisdr.pdf"),
    )

    # -----------------------------------------------------------------------
    # Per-distortion-level bar chart
    # -----------------------------------------------------------------------
    if by_gain:
        print("\nPer-gain-level SI-SDR improvement:")
        for glevel in ["clean", "breakup", "crunch", "cranked"]:
            if glevel in by_gain:
                vals = by_gain[glevel]
                print(f"  {glevel:<15} {np.mean(vals):>+7.2f} dB  (n={len(vals)})")

        plot_bar_chart(
            by_gain,
            "SI-SDR Improvement by Distortion Level",
            "SI-SDR Improvement (dB)",
            str(output_dir / "per_gain_sisdr.pdf"),
        )

    # -----------------------------------------------------------------------
    # Spectrogram comparisons
    # -----------------------------------------------------------------------
    print(f"\nGenerating {min(args.spectrogram_examples, len(spectrogram_candidates))} spectrogram comparisons...")
    for idx, (etype, wet, restored, dry, sr, name) in enumerate(
        spectrogram_candidates[:args.spectrogram_examples]
    ):
        plot_spectrogram_comparison(
            wet, restored, dry, sr,
            title=f"Effect: {etype}",
            output_path=str(output_dir / f"spectrogram_{etype}.pdf"),
        )

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
