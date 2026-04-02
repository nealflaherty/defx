#!/usr/bin/env python3
"""
Launch a SageMaker training job for DemucsDefx.

Before running, set these environment variables or edit the constants below:
    DEFX_S3_BUCKET       - S3 bucket for data and output
    DEFX_SAGEMAKER_ROLE  - SageMaker execution role ARN
    AWS_PROFILE          - AWS CLI profile name
    AWS_REGION           - AWS region (default: us-east-1)

Usage:
    python sagemaker/launch_training.py
    python sagemaker/launch_training.py --epochs 200 --instance ml.g5.xlarge
    python sagemaker/launch_training.py --spot
"""

import argparse
import os

import boto3
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, InputData, Compute
from sagemaker.core.shapes.shapes import OutputDataConfig, StoppingCondition, CheckpointConfig
from sagemaker.core.helper.session_helper import Session

# Configure these for your environment
BUCKET = os.environ.get("DEFX_S3_BUCKET", "YOUR-BUCKET-NAME")
ROLE_ARN = os.environ.get("DEFX_SAGEMAKER_ROLE", "arn:aws:iam::YOUR-ACCOUNT:role/YOUR-ROLE")
REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_PROFILE = os.environ.get("AWS_PROFILE", "default")

# PyTorch 2.5 GPU training DLC for us-east-1
PYTORCH_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker"
)


def main():
    parser = argparse.ArgumentParser(description="Launch DeFX SageMaker training job")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--chunk-samples", type=int, default=44100)
    parser.add_argument("--instance", type=str, default="ml.g5.xlarge")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--unfreeze-decoder-layers", type=int, default=3)
    parser.add_argument("--unfreeze-encoder-layers", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-steps-per-epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    boto_session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    sm_session = Session(boto_session=boto_session)

    s3_dry = f"s3://{BUCKET}/ground_truth/dry"
    s3_wet = f"s3://{BUCKET}/ground_truth/wet"
    s3_output = f"s3://{BUCKET}/output"
    s3_checkpoints = f"s3://{BUCKET}/checkpoints"

    print(f"=== Launching DeFX Training ===")
    print(f"Instance: {args.instance}, Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Data: {s3_dry}, {s3_wet}")

    source_code = SourceCode(
        source_dir="sagemaker",
        requirements="requirements.txt",
        entry_script="train_demucs_defx.py",
    )

    compute = Compute(
        instance_type=args.instance,
        instance_count=1,
        volume_size_in_gb=100,
        enable_managed_spot_training=args.spot,
    )

    stopping = StoppingCondition(max_runtime_in_seconds=28800)
    if args.spot:
        stopping.max_wait_time_in_seconds = 36000

    model_trainer = ModelTrainer(
        training_image=PYTORCH_IMAGE,
        source_code=source_code,
        compute=compute,
        role=ROLE_ARN,
        base_job_name="defx-demucs",
        sagemaker_session=sm_session,
        output_data_config=OutputDataConfig(s3_output_path=s3_output),
        hyperparameters={
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "lr": args.lr,
            "chunk-samples": args.chunk_samples,
            "unfreeze-decoder-layers": args.unfreeze_decoder_layers,
            "unfreeze-encoder-layers": args.unfreeze_encoder_layers,
            "save-every": args.save_every,
            "max-steps-per-epoch": args.max_steps_per_epoch,
            "patience": args.patience,
        },
        stopping_condition=stopping,
        checkpoint_config=CheckpointConfig(s3_uri=s3_checkpoints),
    )

    input_dry = InputData(channel_name="dry", data_source=s3_dry)
    input_wet = InputData(channel_name="wet", data_source=s3_wet)

    print("\nStarting training job...")
    model_trainer.train(input_data_config=[input_dry, input_wet], wait=False)

    training_job = model_trainer._latest_training_job
    job_name = training_job.training_job_name
    print(f"\nTraining job: {job_name}")
    print(f"Console: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job_name}")
    print(f"Logs: aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix {job_name}")
    print(f"Checkpoints: aws s3 ls s3://{BUCKET}/checkpoints/")


if __name__ == "__main__":
    main()
