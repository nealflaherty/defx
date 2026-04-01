# DeFX — Guitar Effect Removal with Deep Learning

DeFX removes distortion and amp effects from guitar recordings, recovering the clean (dry) signal using a fine-tuned [HDemucs](https://github.com/facebookresearch/demucs) backbone.

## How It Works

DeFX takes a pretrained HDemucs music source separation model and fine-tunes it to learn the inverse of guitar amp distortion. Given a recording of a distorted guitar, it predicts what the clean guitar sounded like before the amp.

The model uses:

- Pretrained HDemucs backbone (frozen early layers, unfrozen encoder/decoder layers)
- A learned mixing head that combines the separated sources
- L1-dominant loss function for natural-sounding output
- Training on paired dry/wet audio generated from NAM (Neural Amp Modeler) captures

## Quick Start

### Setup

```bash
source setup.sh
```

This creates a `.venv` virtual environment and installs all dependencies. To reactivate later:

```bash
source .venv/bin/activate
```

### Inference

```bash
python inference.py --input my_distorted_guitar.wav --output clean_guitar.wav
```

You'll need a trained checkpoint at `models/defx/checkpoints/best.pt`.

### Training

DeFX training runs on AWS SageMaker. You'll need:

- An AWS account with SageMaker access
- An S3 bucket for data and model artifacts
- A SageMaker execution role

Set your environment:

```bash
export DEFX_S3_BUCKET=your-bucket-name
export DEFX_SAGEMAKER_ROLE=arn:aws:iam::YOUR-ACCOUNT:role/YOUR-ROLE
export AWS_PROFILE=your-profile
```

#### 1. Generate Ground Truth

Upload dry guitar audio and NAM models to S3:

```bash
aws s3 sync your_guitar_wavs/ s3://$DEFX_S3_BUCKET/idmt/
aws s3 sync models/nam/blackpanel/ s3://$DEFX_S3_BUCKET/nam_models/
```

Launch the ground truth generation job:

```bash
python sagemaker/launch_ground_truth.py
```

This processes each guitar recording through each NAM model, creating paired dry/wet training data.

#### 2. Train

```bash
python sagemaker/launch_training.py --epochs 500 --instance ml.g5.xlarge
```

Monitor progress:

```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix defx-demucs-TIMESTAMP
```

## NAM Captures

The `models/nam/blackpanel/` directory contains 10 Neural Amp Modeler captures of a black panel American tube amp at different settings:

| File                             | Volume | Treble | Bass | Character             |
| -------------------------------- | ------ | ------ | ---- | --------------------- |
| `blackpanel_v2.0_t5.0_b5.0.nam`  | 2.0    | 5.0    | 5.0  | Clean, neutral        |
| `blackpanel_v3.0_t8.0_b3.0.nam`  | 3.0    | 8.0    | 3.0  | Clean, bright         |
| `blackpanel_v4.5_t3.0_b7.0.nam`  | 4.5    | 3.0    | 7.0  | Warm, edge of breakup |
| `blackpanel_v5.0_t5.0_b5.0.nam`  | 5.0    | 5.0    | 5.0  | Neutral breakup       |
| `blackpanel_v5.5_t8.0_b5.0.nam`  | 5.5    | 8.0    | 5.0  | Bright breakup        |
| `blackpanel_v7.0_t3.0_b5.0.nam`  | 7.0    | 3.0    | 5.0  | Dark crunch           |
| `blackpanel_v7.0_t5.0_b5.0.nam`  | 7.0    | 5.0    | 5.0  | Neutral crunch        |
| `blackpanel_v7.0_t8.0_b3.0.nam`  | 7.0    | 8.0    | 3.0  | Bright crunch         |
| `blackpanel_v9.0_t5.0_b5.0.nam`  | 9.0    | 5.0    | 5.0  | Cranked               |
| `blackpanel_v10.0_t5.0_b7.0.nam` | 10.0   | 5.0    | 7.0  | Max drive, bassy      |

These can be used with any NAM-compatible plugin or with the included `effects/nam_loader.py` for batch processing.

## Project Structure

```
defx/
├── setup.sh                  # Create venv and install dependencies
├── inference.py              # Run effect removal on audio files
├── train.ipynb               # Local training notebook
├── models/
│   ├── defx/                 # DeFX model architecture
│   │   ├── demucs_defx.py    # HDemucs backbone + mixing head
│   │   ├── dataset.py        # Dry/wet pair dataset (lazy-loading)
│   │   └── network.py        # STFT U-Net (alternative architecture)
│   └── nam/blackpanel/       # NAM amp captures (.nam files)
├── effects/
│   ├── chain.py              # Effect chain for serial processing
│   └── nam_loader.py         # NAM model loader for batch inference
├── sagemaker/
│   ├── train_demucs_defx.py  # SageMaker training entry point
│   ├── generate_ground_truth.py  # Ground truth generation
│   ├── launch_training.py    # Launch training jobs
│   └── launch_ground_truth.py    # Launch ground truth jobs
└── requirements.txt
```

## License

MIT
