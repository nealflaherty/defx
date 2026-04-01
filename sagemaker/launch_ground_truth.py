#!/usr/bin/env python3
"""
Launch a SageMaker job to generate ground truth dry/wet pairs
from audio files + NAM models.

Upload your data first:
    aws s3 sync your_audio/ s3://YOUR-BUCKET/idmt/
    aws s3 sync models/nam/blackpanel/ s3://YOUR-BUCKET/nam_models/

Usage:
    python sagemaker/launch_ground_truth.py
"""

import os

import boto3
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, InputData, Compute
from sagemaker.core.shapes.shapes import OutputDataConfig, StoppingCondition
from sagemaker.core.helper.session_helper import Session

BUCKET = os.environ.get("DEFX_S3_BUCKET", "YOUR-BUCKET-NAME")
ROLE_ARN = os.environ.get("DEFX_SAGEMAKER_ROLE", "arn:aws:iam::YOUR-ACCOUNT:role/YOUR-ROLE")
REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_PROFILE = os.environ.get("AWS_PROFILE", "default")

PYTORCH_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
)


def main():
    boto_session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    sm_session = Session(boto_session=boto_session)

    s3_audio = f"s3://{BUCKET}/idmt"
    s3_nam = f"s3://{BUCKET}/nam_models"
    s3_output = f"s3://{BUCKET}/ground_truth"

    print(f"=== Ground Truth Generation ===")
    print(f"Audio: {s3_audio}")
    print(f"NAM models: {s3_nam}")
    print(f"Output: {s3_output}")

    source_code = SourceCode(
        source_dir="sagemaker",
        requirements="requirements.txt",
        entry_script="generate_ground_truth.py",
    )

    compute = Compute(instance_type="ml.m5.2xlarge", instance_count=1, volume_size_in_gb=100)

    model_trainer = ModelTrainer(
        training_image=PYTORCH_IMAGE,
        source_code=source_code,
        compute=compute,
        role=ROLE_ARN,
        base_job_name="defx-ground-truth",
        sagemaker_session=sm_session,
        output_data_config=OutputDataConfig(s3_output_path=s3_output),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=28800),
    )

    input_audio = InputData(channel_name="idmt", data_source=s3_audio)
    input_nam = InputData(channel_name="nam_models", data_source=s3_nam)

    print("\nStarting job...")
    model_trainer.train(input_data_config=[input_audio, input_nam], wait=False)

    job_name = model_trainer._latest_training_job.training_job_name
    print(f"\nJob: {job_name}")
    print(f"Logs: aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix {job_name}")


if __name__ == "__main__":
    main()
