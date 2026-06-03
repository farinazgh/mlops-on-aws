#!/usr/bin/env python3
"""
Train a customer churn prediction model with Amazon SageMaker XGBoost.

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
6. Trains an XGBoost model using the built-in SageMaker XGBoost container.
7. Saves the model artifact URI and training metadata to training_output.json.
8. Keeps churn_test.csv locally so the deployment script can optionally evaluate accuracy.

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
from sagemaker.session import Session

load_dotenv()


DATASET_S3_URI = os.getenv(
    "DATASET_S3_URI",
    "s3://sagemaker-sample-files/datasets/tabular/synthetic/churn.txt",
)

DEFAULT_BUCKET = os.getenv("SAGEMAKER_BUCKET")
DEFAULT_ROLE = os.getenv("SAGEMAKER_ROLE")
DEFAULT_TRAINING_INSTANCE_TYPE = os.getenv(
    "SAGEMAKER_TRAINING_INSTANCE_TYPE",
    os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large"),
)
DEFAULT_XGBOOST_VERSION = os.getenv("SAGEMAKER_XGBOOST_VERSION", "1.7-1")

LOCAL_DATASET_PATH = Path("churn.txt")
TRAIN_CSV_PATH = Path("churn_train.csv")
VALIDATION_CSV_PATH = Path("churn_validate.csv")
TEST_CSV_PATH = Path("churn_test.csv")
TRAINING_OUTPUT_JSON_PATH = Path("training_output.json")


def run_command(command: list[str]) -> None:
    """Run a shell command and fail clearly if it does not succeed."""
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


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
    churn_df_shuffled = churn_df.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset_length = len(churn_df_shuffled)

    train_end = int(0.6 * dataset_length)
    validation_end = int(0.8 * dataset_length)

    # pandas-native split, so the result stays as DataFrames.
    # This is safer than np.split here because SageMaker needs CSV files,
    # and only pandas DataFrames have .to_csv().
    #
    #       ┌────────┬────────────┬────────┐
    #       │ Train  │ Validation │ Test   │
    #       └────────┴────────────┴────────┘
    churn_df_train = churn_df_shuffled.iloc[:train_end]
    churn_df_validate = churn_df_shuffled.iloc[train_end:validation_end]
    churn_df_test = churn_df_shuffled.iloc[validation_end:]

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


def upload_training_files(bucket: str, prefix: str) -> tuple[str, str]:
    """Upload train and validation CSV files to S3 and return their S3 URIs."""
    prefix = prefix.strip("/")
    train_s3_uri = f"s3://{bucket}/{prefix}/{TRAIN_CSV_PATH.name}"
    validation_s3_uri = f"s3://{bucket}/{prefix}/{VALIDATION_CSV_PATH.name}"

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
def train_xgboost_model(
    session: Session,
    role: str,
    bucket: str,
    output_prefix: str,
    train_s3_uri: str,
    validation_s3_uri: str,
    training_instance_type: str,
    xgboost_version: str,
) -> Estimator:
    """Create, train, and return a SageMaker built-in XGBoost estimator."""
    s3_input_train = TrainingInput(s3_data=train_s3_uri, content_type="csv")
    s3_input_validate = TrainingInput(s3_data=validation_s3_uri, content_type="csv")

    xgb_image = image_uris.retrieve(
        framework="xgboost",
        region=session.boto_region_name,
        version=xgboost_version,
        image_scope="training",
        instance_type=training_instance_type,
    )

    output_prefix = output_prefix.strip("/")
    xgb = Estimator(
        image_uri=xgb_image,
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=f"s3://{bucket}/{output_prefix}",
        sagemaker_session=session,
    )

    xgb.set_hyperparameters(
        max_depth=5,
        objective="binary:logistic",
        num_round=100,
    )

    xgb.fit(inputs={"train": s3_input_train, "validation": s3_input_validate})
    return xgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a customer churn prediction model using SageMaker XGBoost."
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="S3 bucket to use. Defaults to SAGEMAKER_BUCKET from .env or SageMaker default bucket.",
    )

    parser.add_argument(
        "--role",
        default=DEFAULT_ROLE,
        help="SageMaker execution role ARN. Defaults to SAGEMAKER_ROLE from .env or sagemaker.get_execution_role().",
    )

    parser.add_argument(
        "--training-instance-type",
        default=DEFAULT_TRAINING_INSTANCE_TYPE,
        help="Training instance type. Defaults to SAGEMAKER_TRAINING_INSTANCE_TYPE, SAGEMAKER_INSTANCE_TYPE, or ml.m5.large.",
    )

    parser.add_argument(
        "--data-prefix",
        default="customer-churn-xgboost/data",
        help="S3 prefix where train/validation CSV files will be uploaded.",
    )

    parser.add_argument(
        "--output-prefix",
        default="customer-churn-xgboost/output",
        help="S3 prefix where SageMaker will write the model artifact.",
    )

    parser.add_argument(
        "--xgboost-version",
        default=DEFAULT_XGBOOST_VERSION,
        help="SageMaker built-in XGBoost version. Defaults to 1.7-1.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    session = sagemaker.Session()

    bucket = args.bucket or session.default_bucket()
    role = args.role or sagemaker.get_execution_role()

    print(f"Using bucket: {bucket}")
    print(f"Using role: {role}")
    print(f"Using AWS region: {session.boto_region_name}")
    print(f"Using training instance type: {args.training_instance_type}")

    download_dataset()

    churn_df = load_and_preprocess_data(LOCAL_DATASET_PATH)
    train_df, validation_df, test_df = split_data(churn_df)
    write_csv_files(train_df, validation_df, test_df)

    train_s3_uri, validation_s3_uri = upload_training_files(
        bucket=bucket,
        prefix=args.data_prefix,
    )

    xgb = train_xgboost_model(
        session=session,
        role=role,
        bucket=bucket,
        output_prefix=args.output_prefix,
        train_s3_uri=train_s3_uri,
        validation_s3_uri=validation_s3_uri,
        training_instance_type=args.training_instance_type,
        xgboost_version=args.xgboost_version,
    )

    model_data_s3_uri = xgb.model_data

    output = {
        "training_job_name": xgb.latest_training_job.name,
        "model_data_s3_uri": model_data_s3_uri,
        "training_image_uri": xgb.image_uri,
        "xgboost_version": args.xgboost_version,
        "region": session.boto_region_name,
        "test_csv_path": str(TEST_CSV_PATH),
    }

    TRAINING_OUTPUT_JSON_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nTraining completed successfully.")
    print(f"Model artifact S3 URI: {model_data_s3_uri}")
    print(f"Saved training metadata locally to: {TRAINING_OUTPUT_JSON_PATH}")
    print(f"Saved local test dataset to: {TEST_CSV_PATH}")
    print("\nUse this model artifact URI in the deployment script with:")
    print(
        f"python deploy.py --model-data-s3-uri {model_data_s3_uri} --test-csv {TEST_CSV_PATH}"
    )


if __name__ == "__main__":
    main()
