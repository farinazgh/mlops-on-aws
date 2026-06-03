#!/usr/bin/env python3
"""
Deploy and optionally test an already-trained SageMaker built-in XGBoost
customer churn model.

This script assumes training is already complete.
It does NOT download data, preprocess data, split data, upload training data,
create an Estimator, run fit(), or create a SageMaker Pipeline.

Required input:
- S3 URI of the trained model artifact: model.tar.gz

Typical usage:
    python 02_deploy_test_customer_churn_xgboost_sagemaker.py \
        --model-data-s3-uri s3://your-bucket/path/to/model.tar.gz \
        --endpoint-name churn-xgboost-endpoint

Optional one-row prediction:
    python 02_deploy_test_customer_churn_xgboost_sagemaker.py \
        --model-data-s3-uri s3://your-bucket/path/to/model.tar.gz \
        --sample-csv "1,0,42,3,0,0,1,..." \
        --keep-endpoint

Optional full test-set evaluation:
    python 02_deploy_test_customer_churn_xgboost_sagemaker.py \
        --model-data-s3-uri s3://your-bucket/path/to/model.tar.gz \
        --test-csv churn_test.csv

Important for separate AWS accounts:
- The deployment account must be able to read the S3 model artifact.
- If the model was trained in another account, copy model.tar.gz into this
  account's S3 bucket or configure cross-account S3/KMS permissions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sagemaker
from dotenv import load_dotenv
from sagemaker import image_uris
from sagemaker.deserializers import StringDeserializer
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.session import Session
from sklearn.metrics import accuracy_score

load_dotenv()


DEFAULT_ROLE = os.getenv("SAGEMAKER_ROLE")
DEFAULT_MODEL_DATA_S3_URI = os.getenv("SAGEMAKER_MODEL_DATA_S3_URI")
DEFAULT_ENDPOINT_INSTANCE_TYPE = os.getenv(
    "SAGEMAKER_ENDPOINT_INSTANCE_TYPE", "ml.m5.large"
)
DEFAULT_XGBOOST_VERSION = os.getenv("SAGEMAKER_XGBOOST_VERSION", "1.7-1")
DEFAULT_TEST_CSV_PATH = os.getenv("SAGEMAKER_TEST_CSV_PATH")


def resolve_role(args_role: str | None) -> str:
    """Resolve the SageMaker execution role from CLI, .env, or SageMaker environment."""
    if args_role:
        return args_role

    try:
        return sagemaker.get_execution_role()
    except Exception as exc:
        raise ValueError(
            "SageMaker execution role was not found.\n\n"
            "If you are running locally from PyCharm/terminal, set SAGEMAKER_ROLE "
            "in your .env file or pass --role.\n\n"
            "Example:\n"
            "SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        ) from exc


def create_model_from_artifact(
    session: Session,
    role: str,
    model_data_s3_uri: str,
    endpoint_instance_type: str,
    xgboost_version: str,
) -> Model:
    """Create a SageMaker Model object from an existing model.tar.gz artifact."""
    xgb_inference_image = image_uris.retrieve(
        framework="xgboost",
        region=session.boto_region_name,
        version=xgboost_version,
        image_scope="inference",
        instance_type=endpoint_instance_type,
    )

    return Model(
        image_uri=xgb_inference_image,
        model_data=model_data_s3_uri,
        role=role,
        sagemaker_session=session,
        predictor_cls=Predictor,
    )


def deploy_model(
    model: Model,
    endpoint_instance_type: str,
    endpoint_name: str | None,
) -> Predictor:
    """Deploy the model to a SageMaker real-time endpoint."""
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=endpoint_instance_type,
        endpoint_name=endpoint_name,
    )
    predictor.serializer = CSVSerializer()
    predictor.deserializer = StringDeserializer()
    return predictor


def predict_sample(
    predictor: Predictor,
    sample_csv: str,
) -> str:
    """Run a prediction using one CSV-formatted feature row."""
    return predictor.predict(sample_csv)


def evaluate_model(
    predictor: Predictor,
    test_csv_path: Path,
) -> float:
    """Run inference against the endpoint and return test accuracy."""
    test_df = pd.read_csv(test_csv_path, header=None)
    test_np = test_df.to_numpy()

    # Column 0 is the label because we moved Churn? to the first column.
    actual = test_np[:, 0]

    # All remaining columns are model input features.
    attributes = test_np[:, 1:]

    # SageMaker XGBoost returns one probability per row, separated by newlines.
    results = predictor.predict(attributes)

    if isinstance(results, bytes):
        results = results.decode("utf-8")

    predictions = np.fromstring(str(results), sep="\n")

    if len(predictions) != len(actual):
        raise ValueError(
            "Prediction count does not match test dataset row count. "
            f"Predictions: {len(predictions)}, test rows: {len(actual)}"
        )

    # binary:logistic returns probabilities. Round them to 0/1 class labels.
    predicted_labels = np.round(predictions)

    return accuracy_score(actual, predicted_labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy and optionally test an existing SageMaker XGBoost churn model."
    )

    parser.add_argument(
        "--model-data-s3-uri",
        default=DEFAULT_MODEL_DATA_S3_URI,
        required=DEFAULT_MODEL_DATA_S3_URI is None,
        help="S3 URI of the trained model.tar.gz artifact. Can also come from SAGEMAKER_MODEL_DATA_S3_URI.",
    )
    parser.add_argument(
        "--role",
        default=DEFAULT_ROLE,
        help="SageMaker execution role ARN. Defaults to SAGEMAKER_ROLE from .env or SageMaker environment role.",
    )
    parser.add_argument(
        "--endpoint-instance-type",
        default=DEFAULT_ENDPOINT_INSTANCE_TYPE,
        help="Deployment/inference instance type. Defaults to SAGEMAKER_ENDPOINT_INSTANCE_TYPE or ml.m5.large.",
    )
    parser.add_argument(
        "--endpoint-name",
        default=None,
        help="Optional endpoint name. If omitted, SageMaker generates one.",
    )
    parser.add_argument(
        "--xgboost-version",
        default=DEFAULT_XGBOOST_VERSION,
        help="SageMaker built-in XGBoost version. Use the same version as training. Defaults to 1.7-1.",
    )
    parser.add_argument(
        "--sample-csv",
        default=None,
        help="Optional CSV feature row for a quick prediction test. Do not include the label column.",
    )
    parser.add_argument(
        "--test-csv",
        default=DEFAULT_TEST_CSV_PATH,
        help="Optional local test CSV path generated by train.py. The file should include the label column as the first column.",
    )
    parser.add_argument(
        "--keep-endpoint",
        action="store_true",
        help="Keep the endpoint running after deployment/testing. Warning: this continues to incur cost.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    session = sagemaker.Session()
    role = resolve_role(args.role)

    print(f"Using role: {role}")
    print(f"Using AWS region: {session.boto_region_name}")
    print(f"Using model artifact: {args.model_data_s3_uri}")
    print(f"Using endpoint instance type: {args.endpoint_instance_type}")

    if args.test_csv:
        print(f"Using test CSV: {args.test_csv}")

    model = create_model_from_artifact(
        session=session,
        role=role,
        model_data_s3_uri=args.model_data_s3_uri,
        endpoint_instance_type=args.endpoint_instance_type,
        xgboost_version=args.xgboost_version,
    )

    predictor: Predictor | None = None

    try:
        print("Deploying model to a SageMaker real-time endpoint...")
        predictor = deploy_model(
            model=model,
            endpoint_instance_type=args.endpoint_instance_type,
            endpoint_name=args.endpoint_name,
        )
        print(f"Endpoint is ready: {predictor.endpoint_name}")

        if args.sample_csv:
            print("Running one sample prediction...")
            prediction = predict_sample(
                predictor=predictor,
                sample_csv=args.sample_csv,
            )
            print(f"Prediction result: {prediction}")
        else:
            print(
                "No sample CSV provided. Deployment completed without sample prediction."
            )

        if args.test_csv:
            print("Running inference on the test dataset...")
            accuracy = evaluate_model(
                predictor=predictor,
                test_csv_path=Path(args.test_csv),
            )
            print(f"Test accuracy: {accuracy:.4f}")
        else:
            print("No test CSV provided. Skipping full test-set evaluation.")

    finally:
        if predictor is not None and not args.keep_endpoint:
            print("Deleting endpoint to avoid ongoing charges...")
            predictor.delete_endpoint()
            print("Endpoint deleted.")
        elif predictor is not None:
            print(f"Endpoint kept running: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
