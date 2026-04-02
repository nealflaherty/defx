#!/usr/bin/env python3
"""
SageMaker training entry point for DemucsDefx (wet → dry effect removal).

SageMaker passes data and config via environment variables:
  SM_CHANNEL_DRY   → /opt/ml/input/data/dry/
  SM_CHANNEL_WET   → /opt/ml/input/data/wet/
  SM_MODEL_DIR     → /opt/ml/model/
  SM_HP_*          → hyperparameters

Checkpoints are saved to /opt/ml/checkpoints/ which SageMaker
continuously syncs to S3 so you can monitor progress mid-run.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import auraloss.freq
from models.defx.dataset import DryWetDataset
from models.defx.demucs_defx import DemucsDefx


def find_pairs(dry_dir: str, wet_dir: str) -> list[tuple[str, str]]:
    """Match dry/wet files. Each dry file can have multiple wet versions."""
    dry_dir = Path(dry_dir)
    wet_dir = Path(wet_dir)
    dry_files = sorted(dry_dir.glob("*.wav"))
    wet_files = {p.name: p for p in sorted(wet_dir.glob("*.wav"))}

    pairs = []
    for dry_path in dry_files:
        stem = dry_path.stem
        matched = [wp for wn, wp in wet_files.items()
                   if wn.startswith(stem) and wn.endswith("_wet.wav")]
        if matched:
            for wet_path in matched:
                pairs.append((str(dry_path), str(wet_path)))
        else:
            print(f"  Warning: no wet match for {dry_path.name}")

    print(f"  Found {len(pairs)} pairs from {len(dry_files)} dry files")
    return pairs


def split_pairs(pairs: list[tuple[str, str]], val_fraction: float = 0.1):
    """Split pairs into train/val by dry file to prevent leakage."""
    import hashlib
    # Group by dry file
    by_dry = {}
    for dry_path, wet_path in pairs:
        by_dry.setdefault(dry_path, []).append((dry_path, wet_path))

    dry_files = sorted(by_dry.keys())
    # Deterministic split based on filename hash
    train_pairs, val_pairs = [], []
    for dry_path in dry_files:
        h = int(hashlib.md5(dry_path.encode()).hexdigest(), 16) % 100
        if h < val_fraction * 100:
            val_pairs.extend(by_dry[dry_path])
        else:
            train_pairs.extend(by_dry[dry_path])

    return train_pairs, val_pairs


def train(
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    model_dir: str,
    checkpoint_dir: str,
    epochs: int = 10,
    batch_size: int = 1,
    lr: float = 1e-3,
    chunk_samples: int = 44100,
    unfreeze_decoder_layers: int = 0,
    unfreeze_encoder_layers: int = 0,
    save_every: int = 5,
    max_steps_per_epoch: int = 0,
    patience: int = 0,
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    dataset = DryWetDataset(train_pairs, chunk_samples=chunk_samples, stereo=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2,
    )
    print(f"Train: {len(train_pairs)} pairs, {len(dataset)} chunks/epoch")

    if val_pairs:
        val_dataset = DryWetDataset(val_pairs, chunk_samples=chunk_samples, stereo=True, augment=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        )
        val_steps = min(50, len(val_loader))  # Cap validation at 50 steps
        print(f"Val: {len(val_pairs)} pairs, eval {val_steps} steps")
    else:
        val_loader = None
        val_steps = 0
    if max_steps_per_epoch > 0:
        print(f"Max steps per epoch: {max_steps_per_epoch}")

    print("Loading pretrained HDemucs backbone...")
    model = DemucsDefx(
        freeze_encoder=True,
        unfreeze_decoder_layers=unfreeze_decoder_layers,
        unfreeze_encoder_layers=unfreeze_encoder_layers,
    ).to(device)
    print(f"Total: {model.total_params:,} | Trainable: {model.trainable_params:,}")

    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_lengths=[512, 1024, 2048],
    ).to(device)

    mel_loss_fn = auraloss.freq.MelSTFTLoss(
        sample_rate=44100, fft_size=2048, hop_size=512,
        win_length=2048, n_mels=128,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_stft = 0.0
        epoch_mel = 0.0
        t0 = time.time()

        for step, (wet, dry) in enumerate(loader):
            if max_steps_per_epoch > 0 and step >= max_steps_per_epoch:
                break
            wet = wet.to(device)
            dry = dry.to(device)
            pred = model(wet)
            l1 = F.l1_loss(pred, dry)
            stft = mrstft(pred, dry)
            mel = mel_loss_fn(pred, dry)
            loss = l1 + 0.1 * stft + 0.05 * mel
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_l1 += l1.item()
            epoch_stft += stft.item()
            epoch_mel += mel.item()

        steps_done = min(step + 1, len(loader)) if max_steps_per_epoch > 0 else len(loader)
        avg = epoch_loss / max(steps_done, 1)
        avg_l1 = epoch_l1 / max(steps_done, 1)
        avg_stft = epoch_stft / max(steps_done, 1)
        avg_mel = epoch_mel / max(steps_done, 1)
        elapsed = time.time() - t0

        # Validation loss
        val_loss_str = ""
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vs, (vwet, vdry) in enumerate(val_loader):
                    if vs >= val_steps:
                        break
                    vwet = vwet.to(device)
                    vdry = vdry.to(device)
                    vpred = model(vwet)
                    vl = (
                        F.l1_loss(vpred, vdry)
                        + 0.1 * mrstft(vpred, vdry)
                        + 0.05 * mel_loss_fn(vpred, vdry)
                    )
                    val_loss += vl.item()
            val_avg = val_loss / max(min(vs + 1, val_steps), 1)
            val_loss_str = f"  val={val_avg:.4f}"

        print(f"  Epoch {epoch}/{epochs}  loss={avg:.4f}  l1={avg_l1:.4f}  stft={avg_stft:.4f}  mel={avg_mel:.4f}{val_loss_str}  time={elapsed:.1f}s")

        scheduler.step()

        # Build checkpoint payload — save all trainable state
        ckpt = {
            "epoch": epoch,
            "loss": avg,
            "head_state_dict": model.head.state_dict(),
            "unfreeze_decoder_layers": unfreeze_decoder_layers,
            "unfreeze_encoder_layers": unfreeze_encoder_layers,
        }
        if unfreeze_decoder_layers > 0 or unfreeze_encoder_layers > 0:
            # Save all unfrozen backbone params
            ckpt["backbone_state_dict"] = {
                k: v for k, v in model.backbone.state_dict().items()
                if any(p.requires_grad for p in [v] if isinstance(v, torch.nn.Parameter))
            }
            # Simpler: just save params that require grad
            unfrozen_keys = set()
            for name, param in model.backbone.named_parameters():
                if param.requires_grad:
                    unfrozen_keys.add(name)
            ckpt["backbone_state_dict"] = {
                k: v for k, v in model.backbone.state_dict().items()
                if k in unfrozen_keys
            }

        # Save best to both model_dir and checkpoint_dir
        if avg < best_loss:
            best_loss = avg
            epochs_without_improvement = 0
            torch.save(ckpt, model_dir / "demucs_defx_best.pt")
            torch.save(ckpt, checkpoint_dir / "demucs_defx_best.pt")
            print(f"    ✓ saved best (loss={best_loss:.4f})")
        else:
            epochs_without_improvement += 1

        # Save periodic checkpoints (synced to S3 by SageMaker)
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = checkpoint_dir / f"demucs_defx_epoch_{epoch:03d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"    ✓ checkpoint: {ckpt_path.name}")

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"    ✗ early stopping: no improvement for {patience} epochs")
            break

    # Save final to model_dir
    torch.save(ckpt, model_dir / "demucs_defx_final.pt")
    print(f"\nDone. Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_DRY", "data/dry"))
    parser.add_argument("--wet-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_WET", "data/wet"))
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "model_output"))
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/opt/ml/checkpoints")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("SM_HP_EPOCHS", "10")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("SM_HP_BATCH_SIZE", "1")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("SM_HP_LR", "1e-3")))
    parser.add_argument("--chunk-samples", type=int,
                        default=int(os.environ.get("SM_HP_CHUNK_SAMPLES", "44100")))
    parser.add_argument("--unfreeze-decoder-layers", type=int,
                        default=int(os.environ.get("SM_HP_UNFREEZE_DECODER_LAYERS", "0")))
    parser.add_argument("--unfreeze-encoder-layers", type=int,
                        default=int(os.environ.get("SM_HP_UNFREEZE_ENCODER_LAYERS", "0")))
    parser.add_argument("--save-every", type=int,
                        default=int(os.environ.get("SM_HP_SAVE_EVERY", "5")))
    parser.add_argument("--max-steps-per-epoch", type=int,
                        default=int(os.environ.get("SM_HP_MAX_STEPS_PER_EPOCH", "0")),
                        help="Limit steps per epoch (0=unlimited)")
    parser.add_argument("--patience", type=int,
                        default=int(os.environ.get("SM_HP_PATIENCE", "0")),
                        help="Early stopping patience in epochs (0=disabled)")
    args = parser.parse_args()

    print("=== DeFX Training (HDemucs backbone, stereo) ===")
    print(f"Dry dir:        {args.dry_dir}")
    print(f"Wet dir:        {args.wet_dir}")
    print(f"Model dir:      {args.model_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Unfreeze decoder layers: {args.unfreeze_decoder_layers}")
    print(f"Unfreeze encoder layers: {args.unfreeze_encoder_layers}")

    pairs = find_pairs(args.dry_dir, args.wet_dir)
    if not pairs:
        print("No dry/wet pairs found!")
        sys.exit(1)

    train_pairs, val_pairs = split_pairs(pairs, val_fraction=0.1)
    print(f"Train: {len(train_pairs)} pairs, Val: {len(val_pairs)} pairs\n")

    train(
        train_pairs,
        val_pairs,
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_samples=args.chunk_samples,
        unfreeze_decoder_layers=args.unfreeze_decoder_layers,
        unfreeze_encoder_layers=args.unfreeze_encoder_layers,
        save_every=args.save_every,
        max_steps_per_epoch=args.max_steps_per_epoch,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
