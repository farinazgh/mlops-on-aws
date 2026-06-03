#!/usr/bin/env python3
"""
Create and optionally start a SageMaker Pipeline for customer churn prediction.

This version is intentionally aligned with 01-train_customer_churn_xgboost_sagemaker.py
so PyCharm file comparison shows the real pipeline-specific additions instead of
noise from different naming, ordering, or formatting.

1. Creates a SageMaker session.
2. Downloads the sample customer churn dataset from the public SageMaker sample S3 bucket.
3. Pre-processes the dataset:
   - removes the Phone column
   - converts Area Code to a categorical/object column
   - converts Churn? from True./False. to 1/0
   - moves Churn? to the first column because SageMaker XGBoost expects the label first
   - one-hot encodes categorical columns
4. Splits the data into train/validation/test datasets.
5. Uploads train and validation CSV files to S3.
6. Defines a SageMaker Pipeline with:
   - an XGBoost training step
   - a model registration step
7. Upserts the pipeline.
8. Saves the generated pipeline definition JSON locally.
9. Optionally starts a pipeline execution.

"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sagemaker
from dotenv import load_dotenv
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep

load_dotenv()


DATASET_S3_URI = os.getenv(
    "DATASET_S3_URI",
    "s3://sagemaker-sample-files/datasets/tabular/synthetic/churn.txt",
)

DEFAULT_BUCKET = os.getenv("SAGEMAKER_BUCKET")
DEFAULT_ROLE = os.getenv("SAGEMAKER_ROLE")
DEFAULT_INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")
DEFAULT_PIPELINE_NAME = os.getenv(
    "SAGEMAKER_PIPELINE_NAME",
    "churn-prediction-model-pipeline",
)
DEFAULT_MODEL_PACKAGE_GROUP_NAME = os.getenv(
    "SAGEMAKER_MODEL_PACKAGE_GROUP_NAME",
    "churn-model-group",
)
DEFAULT_S3_PREFIX = os.getenv("SAGEMAKER_S3_PREFIX", "churn-pipeline")
DEFAULT_XGBOOST_VERSION = os.getenv("SAGEMAKER_XGBOOST_VERSION", "1.5-1")

LOCAL_DATASET_PATH = Path("churn.txt")
TRAIN_CSV_PATH = Path("churn_train.csv")
VALIDATION_CSV_PATH = Path("churn_validate.csv")
TEST_CSV_PATH = Path("churn_test.csv")
PIPELINE_DEFINITION_PATH = Path("pipeline_definition.json")


def run_command(command: list[str]) -> None:
    """Run a shell command and fail clearly if it does not succeed."""
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def resolve_role(args_role: str | None) -> str:
    """Resolve the SageMaker execution role from CLI, .env, or SageMaker environment."""
    if args_role:
        return args_role

    try:
        return sagemaker.get_execution_role()
    except Exception as exc:
        raise ValueError(
            "SageMaker execution role was not found.\n\n"
            "Because you are likely running locally from PyCharm/terminal, "
            "set SAGEMAKER_ROLE in your .env file or pass --role.\n\n"
            "Example:\n"
            "SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        ) from exc


def download_dataset() -> None:
    """Download the sample churn dataset from the public SageMaker sample bucket."""
    if LOCAL_DATASET_PATH.exists():
        print(f"Dataset already exists locally: {LOCAL_DATASET_PATH}")
        return

    run_command(["aws", "s3", "cp", DATASET_S3_URI, str(LOCAL_DATASET_PATH)])


#  cleaned data; numeric labels; encoded categories; ML-ready features
def load_and_preprocess_data(dataset_path: Path) -> pd.DataFrame:
    """Load the churn dataset and apply preprocessing."""
    # pandas reads the CSV/text file into memory as a DataFrame.
    # very much like spark.read.csv(...)
    churn_df = pd.read_csv(dataset_path)
    # Phone numbers are useless for prediction.
    # pandas operations usually return a NEW DataFrame. Like Spark transformations: df = df.filter(...)
    churn_df = churn_df.drop("Phone", axis=1)
    # Converts the Area Code column from numeric to categorical/text type. so 415!> 352 category not number
    churn_df["Area Code"] = churn_df["Area Code"].astype(object)
    # Converts False. into 0 and everything else into 1.
    churn_df["Churn?"] = np.where(churn_df["Churn?"] == "False.", 0, 1)
    # The Churn? column is then moved to become the first column in the dataset because the XGBoost algorithm used in this demo requires the target column to appear first.
    churn_df = pd.concat(
        [churn_df["Churn?"], churn_df.drop(["Churn?"], axis=1)],
        axis=1,
    )
    # Converts categorical/text columns into numeric one-hot encoded columns; Spark ML StringIndexer; OneHotEncoder
    churn_df = pd.get_dummies(churn_df)

    return churn_df


def split_data(
    churn_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split the data into train, validation, and test datasets."""
    # Randomly shuffles all rows to evenly distribute churn=1
    # random state 42: Same random order every run.
    churn_df_shuffled = churn_df.sample(frac=1, random_state=42)
    dataset_length = len(churn_df_shuffled)
    # NumPy splits the dataset into 3 parts.
    churn_df_train, churn_df_validate, churn_df_test = np.split(
        churn_df_shuffled,
        [int(0.6 * dataset_length), int(0.8 * dataset_length)],
    )

    return churn_df_train, churn_df_validate, churn_df_test


def write_csv_files(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Write CSV files without headers or indexes for SageMaker XGBoost."""
    train_df.to_csv(TRAIN_CSV_PATH, header=False, index=False)
    validation_df.to_csv(VALIDATION_CSV_PATH, header=False, index=False)
    test_df.to_csv(TEST_CSV_PATH, header=False, index=False)

    print(f"Wrote {TRAIN_CSV_PATH}")
    print(f"Wrote {VALIDATION_CSV_PATH}")
    print(f"Wrote {TEST_CSV_PATH}")


def prepare_local_data(force: bool = False) -> None:
    """Create local train/validation/test CSV files if they do not already exist."""
    files_exist = (
        TRAIN_CSV_PATH.exists()
        and VALIDATION_CSV_PATH.exists()
        and TEST_CSV_PATH.exists()
    )

    if files_exist and not force:
        print("Local CSV files already exist. Skipping dataset preparation.")
        return

    download_dataset()

    churn_df = load_and_preprocess_data(LOCAL_DATASET_PATH)
    train_df, validation_df, test_df = split_data(churn_df)
    write_csv_files(train_df, validation_df, test_df)


def upload_training_files(bucket: str, s3_prefix: str) -> tuple[str, str]:
    """Upload train and validation CSV files to S3 and return their S3 URIs."""
    train_s3_uri = f"s3://{bucket}/{s3_prefix}/input/{TRAIN_CSV_PATH.name}"
    validation_s3_uri = f"s3://{bucket}/{s3_prefix}/input/{VALIDATION_CSV_PATH.name}"

    run_command(["aws", "s3", "cp", str(TRAIN_CSV_PATH), train_s3_uri])
    run_command(["aws", "s3", "cp", str(VALIDATION_CSV_PATH), validation_s3_uri])

    return train_s3_uri, validation_s3_uri


#                                                                                                                                                                              |
# Training Dataset:
# The portion of data used by the model to learn patterns, relationships, and parameters during training.
# The model directly updates its internal weights/trees based on this data.

# Validation Dataset: A separate dataset used during training to evaluate how well the model generalizes to unseen data.
# It helps detect overfitting and tune model/hyperparameters without directly training on it.


# Test Dataset: A final unseen dataset used only after training is complete to measure the model’s real-world performance objectively.
# It provides an unbiased evaluation of the finished model.
#
def build_churn_pipeline(
    session: Session,
    pipeline_session: PipelineSession,
    role: str,
    bucket: str,
    pipeline_name: str,
    model_package_group_name: str,
    train_s3_uri: str,
    validation_s3_uri: str,
    instance_type: str,
    s3_prefix: str,
    xgboost_version: str,
) -> Pipeline:
    """Build and return the SageMaker Pipeline object."""
    s3_input_train = TrainingInput(
        s3_data=train_s3_uri,
        content_type="csv",
    )

    s3_input_validate = TrainingInput(
        s3_data=validation_s3_uri,
        content_type="csv",
    )

    xgb_image = image_uris.retrieve(
        framework="xgboost",
        region=session.boto_region_name,
        version=xgboost_version,
        image_scope="training",
        instance_type=instance_type,
    )

    xgb = Estimator(
        image_uri=xgb_image,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        output_path=f"s3://{bucket}/{s3_prefix}/output",
        sagemaker_session=pipeline_session,
    )

    xgb.set_hyperparameters(
        max_depth=5,
        objective="binary:logistic",
        num_round=100,
    )

    churn_training_step = TrainingStep(
        name="ChurnTrainingStep",
        step_args=xgb.fit(
            inputs={
                "train": s3_input_train,
                "validation": s3_input_validate,
            }
        ),
    )

    model = Model(
        image_uri=xgb_image,
        model_data=churn_training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    register_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[instance_type],
        transform_instances=[instance_type],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
    )

    register_model_step = ModelStep(
        name="ChurnRegisterModel",
        step_args=register_args,
    )

    return Pipeline(
        name=pipeline_name,
        steps=[churn_training_step, register_model_step],
        sagemaker_session=pipeline_session,
    )


def save_pipeline_definition(pipeline_description: dict) -> None:
    """Save the generated pipeline definition JSON to a local file."""
    pipeline_definition = json.loads(pipeline_description["PipelineDefinition"])

    PIPELINE_DEFINITION_PATH.write_text(
        json.dumps(
            pipeline_definition,
            sort_keys=True,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(f"Saved pipeline definition to {PIPELINE_DEFINITION_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a SageMaker Pipeline for customer churn prediction."
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="S3 bucket to use. Defaults to SAGEMAKER_BUCKET from .env or SageMaker default bucket.",
    )

    parser.add_argument(
        "--role",
        default=DEFAULT_ROLE,
        help="SageMaker execution role ARN. Defaults to SAGEMAKER_ROLE from .env or SageMaker environment role.",
    )

    parser.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help="Training instance type. Defaults to SAGEMAKER_INSTANCE_TYPE from .env or ml.m5.large.",
    )

    parser.add_argument(
        "--pipeline-name",
        default=DEFAULT_PIPELINE_NAME,
        help="Name of the SageMaker Pipeline.",
    )

    parser.add_argument(
        "--model-package-group-name",
        default=DEFAULT_MODEL_PACKAGE_GROUP_NAME,
        help="Name of the SageMaker Model Package Group.",
    )

    parser.add_argument(
        "--s3-prefix",
        default=DEFAULT_S3_PREFIX,
        help="S3 prefix for pipeline inputs and outputs.",
    )

    parser.add_argument(
        "--xgboost-version",
        default=DEFAULT_XGBOOST_VERSION,
        help="SageMaker built-in XGBoost container version.",
    )

    parser.add_argument(
        "--force-prepare-data",
        action="store_true",
        help="Recreate local train/validation/test CSV files even if they already exist.",
    )

    parser.add_argument(
        "--skip-data-preparation",
        action="store_true",
        help="Skip local data preparation and assume churn_train.csv and churn_validate.csv already exist.",
    )

    parser.add_argument(
        "--start-execution",
        action="store_true",
        help="Start a pipeline execution after upserting the pipeline.",
    )

    parser.add_argument(
        "--execution-display-name",
        default="first-pipeline-execution",
        help="Display name for the pipeline execution.",
    )

    parser.add_argument(
        "--execution-description",
        default="Starting from standalone Python script",
        help="Description for the pipeline execution.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    session = sagemaker.Session()

    bucket = args.bucket or session.default_bucket()
    role = resolve_role(args.role)
    instance_type = args.instance_type
    pipeline_name = args.pipeline_name
    model_package_group_name = args.model_package_group_name
    s3_prefix = args.s3_prefix
    xgboost_version = args.xgboost_version

    pipeline_session = PipelineSession(default_bucket=bucket)

    print(f"Using bucket: {bucket}")
    print(f"Using role: {role}")
    print(f"Using AWS region: {session.boto_region_name}")
    print(f"Using training instance type: {instance_type}")
    print(f"Using pipeline name: {pipeline_name}")
    print(f"Using model package group: {model_package_group_name}")
    print(f"Using S3 prefix: {s3_prefix}")
    print(f"Using XGBoost version: {xgboost_version}")

    if not args.skip_data_preparation:
        prepare_local_data(force=args.force_prepare_data)

    train_s3_uri, validation_s3_uri = upload_training_files(
        bucket=bucket,
        s3_prefix=s3_prefix,
    )

    pipeline = build_churn_pipeline(
        session=session,
        pipeline_session=pipeline_session,
        role=role,
        bucket=bucket,
        pipeline_name=pipeline_name,
        model_package_group_name=model_package_group_name,
        train_s3_uri=train_s3_uri,
        validation_s3_uri=validation_s3_uri,
        instance_type=instance_type,
        s3_prefix=s3_prefix,
        xgboost_version=xgboost_version,
    )

    print("Upserting SageMaker Pipeline...")
    pipeline.upsert(role_arn=role)

    pipeline_description = pipeline.describe()
    save_pipeline_definition(pipeline_description)

    print("Pipeline upsert completed successfully.")

    if args.start_execution:
        print("Starting pipeline execution...")
        execution = pipeline.start(
            execution_display_name=args.execution_display_name,
            execution_description=args.execution_description,
        )
        print(f"Pipeline execution started: {execution.arn}")
    else:
        print("Pipeline execution not started.")
        print("Run again with --start-execution to start the pipeline.")


if __name__ == "__main__":
    main()
