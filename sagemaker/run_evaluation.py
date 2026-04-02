#!/usr/bin/env python3
"""
SageMaker entry point for DeFX evaluation.

Wraps evaluate.py with SageMaker channel paths:
  SM_CHANNEL_DRY        → /opt/ml/input/data/dry/
  SM_CHANNEL_WET        → /opt/ml/input/data/wet/
  SM_CHANNEL_CHECKPOINT  → /opt/ml/input/data/checkpoint/
  SM_MODEL_DIR          → /opt/ml/model/  (results go here)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int,
                        default=int(os.environ.get("SM_HP_MAX_PAIRS", "500")))
    args = parser.parse_args()

    dry_dir = os.environ.get("SM_CHANNEL_DRY", "/opt/ml/input/data/dry")
    wet_dir = os.environ.get("SM_CHANNEL_WET", "/opt/ml/input/data/wet")
    ckpt_dir = os.environ.get("SM_CHANNEL_CHECKPOINT", "/opt/ml/input/data/checkpoint")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    # Find the best checkpoint
    ckpt_path = Path(ckpt_dir) / "demucs_defx_best.pt"
    if not ckpt_path.exists():
        # Try any .pt file
        pt_files = list(Path(ckpt_dir).glob("*.pt"))
        if pt_files:
            ckpt_path = pt_files[0]
        else:
            print(f"No checkpoint found in {ckpt_dir}")
            sys.exit(1)

    output_dir = Path(model_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dry dir:    {dry_dir}")
    print(f"Wet dir:    {wet_dir}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output:     {output_dir}")
    print(f"Max pairs:  {args.max_pairs}")

    # Install evaluate.py dependencies (matplotlib, scipy)
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "scipy"],
                   check=True, capture_output=True)

    # evaluate.py is in the same source_dir, deployed alongside this script
    eval_script = Path(__file__).parent / "evaluate.py"
    if not eval_script.exists():
        # Fallback: check current working directory
        eval_script = Path("evaluate.py")
    if not eval_script.exists():
        print(f"evaluate.py not found")
        sys.exit(1)

    cmd = [
        sys.executable, str(eval_script),
        "--dry-dir", dry_dir,
        "--wet-dir", wet_dir,
        "--checkpoint", str(ckpt_path),
        "--output-dir", str(output_dir),
        "--max-pairs", str(args.max_pairs),
        "--device", "cpu",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
