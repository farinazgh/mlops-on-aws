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
7. Optionally deploys the trained model to a SageMaker real-time endpoint.
8. Runs inference on the test dataset and calculates accuracy.
9. Deletes the endpoint by default to avoid ongoing charges.

"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sagemaker
from dotenv import load_dotenv
from sagemaker import image_uris
from sagemaker.deserializers import StringDeserializer
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.session import Session
from sklearn.metrics import accuracy_score

load_dotenv()


DATASET_S3_URI = os.getenv(
    "DATASET_S3_URI",
    "s3://sagemaker-sample-files/datasets/tabular/synthetic/churn.txt",
)

DEFAULT_BUCKET = os.getenv("SAGEMAKER_BUCKET")
DEFAULT_ROLE = os.getenv("SAGEMAKER_ROLE")
DEFAULT_INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")
DEFAULT_ENDPOINT_INSTANCE_TYPE = os.getenv(
    "SAGEMAKER_ENDPOINT_INSTANCE_TYPE",
    "ml.m5.large",
)

LOCAL_DATASET_PATH = Path("churn.txt")
TRAIN_CSV_PATH = Path("churn_train.csv")
VALIDATION_CSV_PATH = Path("churn_validate.csv")
TEST_CSV_PATH = Path("churn_test.csv")


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
    churn_df_shuffled = churn_df.sample(frac=1, random_state=42)
    dataset_length = len(churn_df_shuffled)
    # NumPy splits the dataset into 3 parts.
    #                One split operation
    #                        │
    #                        ▼
    #       ┌────────┬────────────┬────────┐
    #       │ Train  │ Validation │ Test   │
    #       └────────┴────────────┴────────┘
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


def upload_training_files(bucket: str) -> tuple[str, str]:
    """Upload train and validation CSV files to S3 and return their S3 URIs."""
    train_s3_uri = f"s3://{bucket}/{TRAIN_CSV_PATH.name}"
    validation_s3_uri = f"s3://{bucket}/{VALIDATION_CSV_PATH.name}"

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
    train_s3_uri: str,
    validation_s3_uri: str,
    instance_type: str,
) -> Estimator:
    """Create, train, and return a SageMaker built-in XGBoost estimator."""
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
        version="1.7-1",
        image_scope="training",
        instance_type=instance_type,
    )

    xgb = Estimator(
        image_uri=xgb_image,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        output_path=f"s3://{bucket}/output",
        sagemaker_session=session,
    )

    xgb.set_hyperparameters(
        max_depth=5,
        objective="binary:logistic",
        num_round=100,
    )

    xgb.fit(
        inputs={
            "train": s3_input_train,
            "validation": s3_input_validate,
        }
    )

    return xgb


def deploy_model(
    estimator: Estimator,
    endpoint_instance_type: str,
) -> Predictor:
    """Deploy the trained estimator to a SageMaker real-time endpoint."""
    return estimator.deploy(
        initial_instance_count=1,
        instance_type=endpoint_instance_type,
        serializer=CSVSerializer(),
        deserializer=StringDeserializer(),
    )


def evaluate_model(
    predictor: Predictor,
    test_df: pd.DataFrame,
) -> float:
    """Run inference against the endpoint and return test accuracy."""
    test_np = test_df.to_numpy()

    # Column 0 is the label because we moved Churn? to the first column.
    actual = test_np[:, 0]

    # All remaining columns are model input features.
    attributes = test_np[:, 1:]

    # SageMaker XGBoost returns one probability per row, separated by newlines.
    results = predictor.predict(attributes)
    predictions = np.fromstring(results, sep="\n")

    # binary:logistic returns probabilities. Round them to 0/1 class labels.
    predicted_labels = np.round(predictions)

    return accuracy_score(actual, predicted_labels)


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
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help="Training instance type. Defaults to SAGEMAKER_INSTANCE_TYPE from .env or ml.m5.large.",
    )

    parser.add_argument(
        "--endpoint-instance-type",
        default=DEFAULT_ENDPOINT_INSTANCE_TYPE,
        help="Endpoint instance type. Defaults to SAGEMAKER_ENDPOINT_INSTANCE_TYPE from .env or ml.m5.large.",
    )

    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy the trained model, run test inference, calculate accuracy, and then delete the endpoint.",
    )

    parser.add_argument(
        "--keep-endpoint",
        action="store_true",
        help="Keep the deployed endpoint running after evaluation. Warning: this continues to incur cost.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    session = sagemaker.Session()

    bucket = args.bucket or session.default_bucket()
    role = args.role or sagemaker.get_execution_role()
    instance_type = args.instance_type
    endpoint_instance_type = args.endpoint_instance_type

    print(f"Using bucket: {bucket}")
    print(f"Using role: {role}")
    print(f"Using AWS region: {session.boto_region_name}")
    print(f"Using training instance type: {instance_type}")
    print(f"Using endpoint instance type: {endpoint_instance_type}")

    download_dataset()

    churn_df = load_and_preprocess_data(LOCAL_DATASET_PATH)
    train_df, validation_df, test_df = split_data(churn_df)
    write_csv_files(train_df, validation_df, test_df)

    train_s3_uri, validation_s3_uri = upload_training_files(bucket)

    xgb = train_xgboost_model(
        session=session,
        role=role,
        bucket=bucket,
        train_s3_uri=train_s3_uri,
        validation_s3_uri=validation_s3_uri,
        instance_type=instance_type,
    )

    print("Training completed successfully.")

    if not args.deploy:
        print(
            "Skipping deployment. Run again with --deploy to create an endpoint and test inference."
        )
        return

    predictor: Predictor | None = None

    try:
        print("Deploying trained model to a real-time SageMaker endpoint...")
        predictor = deploy_model(
            estimator=xgb,
            endpoint_instance_type=endpoint_instance_type,
        )

        print("Running inference on the test dataset...")
        accuracy = evaluate_model(
            predictor=predictor,
            test_df=test_df,
        )
        print(f"Test accuracy: {accuracy:.4f}")

    finally:
        if predictor is not None and not args.keep_endpoint:
            print("Deleting endpoint to avoid ongoing charges...")
            predictor.delete_endpoint()
            print("Endpoint deleted.")
        elif predictor is not None:
            print(f"Endpoint kept running: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
