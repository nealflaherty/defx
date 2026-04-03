#!/usr/bin/env python3
"""
Capture and train NAM models from a VST3 amp plugin.

This is an optional step for users who have VST3 amp plugins installed.
It runs the official NAM input signal through the plugin at various
settings, then trains NAM WaveNet models from each capture.

The resulting .nam files go into models/nam/ and can be used for
ground truth generation.

Requires:
    - A VST3 amp plugin installed on your system
    - The official NAM input signal (input.wav)
    - pedalboard and neural-amp-modeler packages

Usage:
    # List available VST3 plugins
    python capture_amp.py --list-plugins

    # List parameters for a plugin
    python capture_amp.py --plugin /path/to/amp.vst3 --list-params

    # Capture + train from a config file
    python capture_amp.py --config amp_config.json

    # Capture + train a single setting
    python capture_amp.py --plugin /path/to/amp.vst3 --name my_amp \\
        --param volume=7.0 --param treble=5.0 --param bass=5.0

    # Capture only (no NAM training)
    python capture_amp.py --plugin /path/to/amp.vst3 --name my_amp \\
        --param volume=7.0 --capture-only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pedalboard import Pedalboard, load_plugin
from pedalboard.io import AudioFile

NAM_INPUT_URL = "https://drive.google.com/file/d/1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G"
NAM_INPUT = Path("nam_input.wav")
CAPTURE_DIR = Path("captures")
MODEL_DIR = Path("models/nam")


def list_plugins():
    """List all VST3 plugins found on the system."""
    import platform
    search_dirs = []
    system = platform.system()
    if system == "Darwin":
        search_dirs = [
            Path("/Library/Audio/Plug-Ins/VST3"),
            Path.home() / "Library/Audio/Plug-Ins/VST3",
        ]
    elif system == "Linux":
        search_dirs = [Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3"), Path.home() / ".vst3"]
    elif system == "Windows":
        search_dirs = [Path("C:/Program Files/Common Files/VST3")]

    plugins = []
    for d in search_dirs:
        if d.exists():
            plugins.extend(sorted(d.glob("*.vst3")))
    return plugins


def list_params(plugin_path: str):
    """List all parameters for a VST3 plugin."""
    plugin = load_plugin(plugin_path)
    print(f"Plugin: {Path(plugin_path).stem}")
    print(f"Parameters ({len(plugin.parameters)}):")
    for name, param in sorted(plugin.parameters.items()):
        print(f"  {name}: {param}")


def ensure_nam_input():
    """Download the NAM input signal if it doesn't exist."""
    if NAM_INPUT.exists():
        return
    print(f"Downloading NAM input signal...")
    import gdown
    gdown.download(id="1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G", output=str(NAM_INPUT), quiet=False)
    if not NAM_INPUT.exists():
        print(f"Download failed. Get it manually from: {NAM_INPUT_URL}")
        sys.exit(1)
    print(f"Saved: {NAM_INPUT}")


def capture(plugin_path: str, params: dict, name: str, tag: str) -> Path:
    """Run NAM input through the plugin, return output path."""
    out_path = CAPTURE_DIR / name / f"{tag}.wav"
    if out_path.exists():
        print(f"  [cached] {tag}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plugin = load_plugin(plugin_path)

    # Set parameters (case-insensitive)
    available = {k.lower(): k for k in plugin.parameters.keys()}
    for k, v in params.items():
        key = available.get(k.lower())
        if key:
            setattr(plugin, key, v)
        else:
            print(f"  warning: param '{k}' not found, available: {list(plugin.parameters.keys())}")

    board = Pedalboard([plugin])

    with AudioFile(str(NAM_INPUT)) as f:
        audio = f.read(f.frames)
        sr = f.samplerate
        ch = f.num_channels

    wet = board(audio, sr)
    with AudioFile(str(out_path), "w", sr, ch) as o:
        o.write(wet)

    peak = float(np.max(np.abs(wet)))
    print(f"  [captured] {tag} (peak={peak:.4f})")
    return out_path


def train_nam(capture_path: Path, name: str, epochs: int, architecture: str) -> str | None:
    """Train a NAM model from a capture."""
    from nam.train.core import train

    tag = capture_path.stem
    export_dir = MODEL_DIR / name
    nam_file = export_dir / f"{tag}.nam"

    if nam_file.exists():
        print(f"  [cached] {nam_file}")
        return str(nam_file)

    train_dir = export_dir / tag / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Training {tag} ({epochs} epochs, {architecture})...")
    result = train(
        input_path=str(NAM_INPUT),
        output_path=str(capture_path),
        train_path=str(train_dir),
        epochs=epochs,
        model_type="WaveNet",
        architecture=architecture,
        lr=0.004,
        lr_decay=0.007,
        batch_size=16,
        ignore_checks=True,
        silent=False,
        modelname=tag,
        save_plot=True,
    )

    if result and result.model is not None:
        result.model.net.export(export_dir, basename=tag)
        esr = getattr(result.metadata, "validation_esr", None) if result.metadata else None
        if esr is not None:
            print(f"  ESR: {esr:.6f}")
        print(f"  Exported: {nam_file}")
        return str(nam_file)
    else:
        print(f"  FAILED: {tag}")
        return None


def make_tag(params: dict) -> str:
    """Create a filename-safe tag from parameter values."""
    parts = [f"{k}{v}" for k, v in sorted(params.items())]
    return "_".join(parts).replace(".", "p")


def run_from_config(config_path: str, capture_only: bool, epochs: int, architecture: str):
    """Run captures and training from a JSON config file."""
    with open(config_path) as f:
        config = json.load(f)

    plugin_path = config["plugin"]
    name = config["name"]
    fixed_params = config.get("fixed_params", {})
    settings = config["settings"]

    print(f"=== {name} ===")
    print(f"Plugin: {plugin_path}")
    print(f"Settings: {len(settings)}")
    print(f"Fixed params: {fixed_params}\n")

    results = []
    for i, setting in enumerate(settings):
        params = {**fixed_params, **setting}
        tag = f"{name}_{make_tag(setting)}"
        print(f"[{i+1}/{len(settings)}] {tag}")

        cap_path = capture(plugin_path, params, name, tag)

        nam_file = None
        if not capture_only:
            nam_file = train_nam(cap_path, name, epochs, architecture)

        results.append({"tag": tag, "params": setting, "nam_file": nam_file})
        print()

    # Summary
    summary_path = MODEL_DIR / name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    trained = sum(1 for r in results if r.get("nam_file"))
    print(f"Done. {trained}/{len(settings)} models trained.")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Capture and train NAM models from VST3 amps")
    parser.add_argument("--list-plugins", action="store_true", help="List available VST3 plugins")
    parser.add_argument("--list-params", action="store_true", help="List plugin parameters")
    parser.add_argument("--plugin", type=str, help="Path to VST3 plugin")
    parser.add_argument("--name", type=str, help="Name for this amp (used in output paths)")
    parser.add_argument("--config", type=str, help="JSON config file for batch capture")
    parser.add_argument("--param", type=str, action="append", help="Parameter as name=value (repeatable)")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--architecture", type=str, default="lite",
                        choices=["nano", "feather", "lite", "standard"])
    args = parser.parse_args()

    if args.list_plugins:
        plugins = list_plugins()
        if plugins:
            print("Found VST3 plugins:")
            for p in plugins:
                print(f"  {p}")
        else:
            print("No VST3 plugins found")
        return

    if args.list_params:
        if not args.plugin:
            print("--plugin required with --list-params")
            sys.exit(1)
        list_params(args.plugin)
        return

    ensure_nam_input()

    if args.config:
        run_from_config(args.config, args.capture_only, args.epochs, args.architecture)
        return

    # Single capture mode
    if not args.plugin or not args.name:
        print("Provide --plugin and --name, or use --config for batch mode")
        sys.exit(1)

    params = {}
    if args.param:
        for entry in args.param:
            k, v = entry.split("=", 1)
            params[k] = float(v)

    tag = f"{args.name}_{make_tag(params)}" if params else args.name
    print(f"Capturing: {tag}")
    cap_path = capture(args.plugin, params, args.name, tag)

    if not args.capture_only:
        train_nam(cap_path, args.name, args.epochs, args.architecture)


if __name__ == "__main__":
    main()
