#!/usr/bin/env python3
"""
Launch a SageMaker processing/training job to run DeFX evaluation.

Runs evaluate.py on a CPU instance with the ground truth data and
a trained checkpoint. Results (figures, CSV) are written to S3.

Usage:
    python sagemaker/launch_evaluation.py
    python sagemaker/launch_evaluation.py --max-pairs 1000
"""

import argparse
import os

import boto3
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, InputData, Compute
from sagemaker.core.shapes.shapes import OutputDataConfig, StoppingCondition
from sagemaker.core.helper.session_helper import Session

BUCKET = os.environ.get("DEFX_S3_BUCKET", "defx-629711664886")
ROLE_ARN = os.environ.get("DEFX_SAGEMAKER_ROLE", "")
REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_PROFILE = os.environ.get("AWS_PROFILE", "defx-trainer")

PYTORCH_IMAGE = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
    "pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
)


def main():
    parser = argparse.ArgumentParser(description="Launch DeFX evaluation on SageMaker")
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--instance", type=str, default="ml.m5.2xlarge")
    args = parser.parse_args()

    if not ROLE_ARN:
        # Derive from account ID
        boto_session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
        account_id = boto_session.client("sts").get_caller_identity()["Account"]
        role_arn = f"arn:aws:iam::{account_id}:role/DefxSageMakerRole"
    else:
        boto_session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
        role_arn = ROLE_ARN

    sm_session = Session(boto_session=boto_session)

    print(f"=== Launching DeFX Evaluation ===")
    print(f"Instance: {args.instance}, Max pairs: {args.max_pairs}")

    source_code = SourceCode(
        source_dir="sagemaker",
        requirements="requirements.txt",
        entry_script="run_evaluation.py",
    )

    trainer = ModelTrainer(
        training_image=PYTORCH_IMAGE,
        source_code=source_code,
        compute=Compute(instance_type=args.instance, instance_count=1, volume_size_in_gb=100),
        role=role_arn,
        base_job_name="defx-eval",
        sagemaker_session=sm_session,
        output_data_config=OutputDataConfig(s3_output_path=f"s3://{BUCKET}/eval_output"),
        hyperparameters={
            "max-pairs": args.max_pairs,
        },
        stopping_condition=StoppingCondition(max_runtime_in_seconds=14400),
    )

    trainer.train(
        input_data_config=[
            InputData(channel_name="dry", data_source=f"s3://{BUCKET}/ground_truth/dry"),
            InputData(channel_name="wet", data_source=f"s3://{BUCKET}/ground_truth/wet"),
            InputData(channel_name="checkpoint", data_source=f"s3://{BUCKET}/checkpoints"),
        ],
        wait=False,
    )

    job_name = trainer._latest_training_job.training_job_name
    print(f"\nEval job: {job_name}")
    print(f"Results will be at: s3://{BUCKET}/eval_output/")
    print(f"Logs: aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix {job_name} --region {REGION} --profile {AWS_PROFILE}")


if __name__ == "__main__":
    main()
