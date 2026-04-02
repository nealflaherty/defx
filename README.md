# DeFX — Guitar Effect Removal with Deep Learning

DeFX removes distortion and amp effects from guitar recordings, recovering the clean (dry) signal using a fine-tuned [HDemucs](https://github.com/facebookresearch/demucs) backbone.

## How It Works

DeFX takes a pretrained HDemucs music source separation model and fine-tunes it to learn the inverse of guitar amp distortion. Given a recording of a distorted guitar, it predicts what the clean guitar sounded like before the amp.

The training pipeline:

1. **Capture** — Run a standardized test signal through VST3 amp plugins at various settings
2. **Model** — Train [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) (NAM) WaveNet models from each capture
3. **Ground Truth** — Process thousands of clean guitar recordings through the NAM models to create paired dry/wet training data
4. **Train** — Fine-tune HDemucs on the paired data using SageMaker

The model uses:

- Pretrained HDemucs backbone with selectively unfrozen encoder/decoder layers
- A learned mixing head that combines the separated sources
- L1-dominant loss function for natural-sounding output

## Quick Start

### Inference

```bash
pip install -r requirements.txt
python inference.py --input my_distorted_guitar.wav --output clean_guitar.wav
```

You'll need a trained checkpoint at `models/defx/checkpoints/best.pt`.

### Full Training Pipeline

The easiest way to run the full pipeline is the training notebook:

```bash
pip install -r requirements.txt
jupyter notebook train.ipynb
```

The notebook walks through every step — AWS setup, data download, ground truth generation, training, and evaluation. Just create a `.env` file with your AWS profile (optional) and run each cell.

```
# .env (optional — uses default AWS profile if not set)
AWS_PROFILE=your-profile
AWS_REGION=us-east-1
```

## Capturing Your Own Amp Models

If you have VST3 amp plugins installed, you can create NAM captures to expand the training data. This is optional — the repo includes pre-made captures of three amp types.

### Prerequisites

- A VST3 amp plugin installed on your system
- The official [NAM input signal](https://drive.google.com/file/d/1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G) saved as `nam_input.wav`
- `pedalboard` and `neural-amp-modeler` packages

### Discovery

```bash
# List VST3 plugins on your system
python capture_amp.py --list-plugins

# List parameters for a specific plugin
python capture_amp.py --plugin /path/to/amp.vst3 --list-params
```

### Single Capture

```bash
python capture_amp.py --plugin /path/to/amp.vst3 --name my_amp \
    --param volume=7.0 --param treble=5.0 --param bass=5.0
```

### Batch Capture from Config

Create a JSON config that defines the plugin, fixed parameters, and a sweep of settings:

```bash
python capture_amp.py --config amp_configs/blackpanel.json
```

Capture only (fast) then train separately:

```bash
python capture_amp.py --config amp_configs/chime.json --capture-only
python capture_amp.py --config amp_configs/chime.json --train-only --epochs 50
```

### Config File Format

```json
{
  "name": "my_amp",
  "description": "Description of the amp character",
  "plugin": "/path/to/plugin.vst3",
  "fixed_params": {
    "reverb": false,
    "output": 5.0,
    "power": true
  },
  "settings": [
    { "volume": 2.0, "treble": 5.0, "bass": 5.0 },
    { "volume": 7.0, "treble": 5.0, "bass": 5.0 },
    { "volume": 10.0, "treble": 5.0, "bass": 5.0 }
  ]
}
```

Each entry in `settings` is combined with `fixed_params` and run through the plugin. The resulting `.nam` files go into `models/nam/{name}/` and can be uploaded to S3 for ground truth generation.

## Included NAM Captures

Three amp types are included, each with 10 settings from clean to full drive:

### Black Panel (American tube amp)

Warm cleans, smooth breakup, classic American character.

### Chime (British Class A)

Chimey cleans, harmonically rich overdrive, top-boost character.

### Plexi (British high gain)

Aggressive midrange, saturated lead tones, classic rock character.

## Project Structure

```
defx/
├── inference.py              # Run effect removal on audio files
├── capture_amp.py            # Capture and train NAM models from VST3 plugins
├── train.ipynb               # Interactive training pipeline notebook
├── amp_configs/              # Amp sweep configurations
│   ├── blackpanel.json       # American tube amp settings
│   ├── chime.json            # British Class A settings
│   └── plexi.json            # British high gain settings
├── models/
│   ├── defx/                 # DeFX model architecture
│   │   ├── demucs_defx.py    # HDemucs backbone + mixing head
│   │   ├── dataset.py        # Dry/wet pair dataset (lazy-loading)
│   │   └── network.py        # STFT U-Net (alternative architecture)
│   └── nam/                  # NAM amp captures (.nam files)
│       └── blackpanel/       # Pre-trained black panel captures
├── effects/
│   ├── chain.py              # Effect chain for serial processing
│   └── nam_loader.py         # NAM model loader for batch inference
├── sagemaker/
│   ├── train_demucs_defx.py  # SageMaker training entry point
│   ├── generate_ground_truth.py  # Ground truth generation
│   ├── launch_training.py    # Launch training jobs
│   └── launch_ground_truth.py    # Launch ground truth jobs
├── examples/
│   ├── test_wet.wav          # Example distorted guitar
│   └── test_dry.wav          # Example clean reference
└── requirements.txt
```

## Training Details

The training uses AWS SageMaker with a PyTorch Deep Learning Container. Key settings:

- **Instance**: `ml.g5.xlarge` (A10G GPU) or `ml.g4dn.xlarge` (T4, cheaper)
- **Architecture**: HDemucs with 3 encoder + 3 decoder layers unfrozen
- **Loss**: L1 (primary) + 0.1 × Multi-Resolution STFT + 0.05 × Mel STFT
- **Optimizer**: AdamW with cosine annealing LR schedule
- **Data**: ~12,000 dry/wet pairs from IDMT-SMT-GUITAR_V2 dataset × 10 NAM settings

Training data is the [IDMT-SMT-GUITAR_V2](https://zenodo.org/records/7544110) dataset, which contains isolated guitar recordings across multiple instruments, playing styles, and genres.

## License

MIT
