import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path so we can import preprocessing
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (
    validate_dataframe,
    clean_data,
    encode_categoricals,
    check_data_quality,
)


def get_config():
    with open("./configs/config.yaml", "r") as file:
        return yaml.safe_load(file)


def load_data(url):
    """Load dataset from URL."""
    print(f"Loading data from {url}...")
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def train_model():
    """Full training pipeline. Returns metrics dictionary."""
    config = get_config()

    # Load
    df = load_data(config["data_url_raw"])

    # Drop Employee Number column
    if "EmployeeNumber" in df.columns:
        df = df.drop(columns=["EmployeeNumber"])

    # Validate
    required = (
        config["numeric_columns"] + config["categorical_columns"] + [config["target"]]
    )
    validate_dataframe(df, required, config["target"])

    # Data quality check
    quality = check_data_quality(df, config["numeric_columns"])
    print(
        f"Data quality: {quality['total_nulls']} nulls, {quality['duplicate_rows']} duplicates"
    )

    # Clean
    df = clean_data(df, config["numeric_columns"], config["categorical_columns"])

    # Encode
    df = encode_categoricals(df, config["categorical_columns"])

    # Split
    X = df.drop(columns=[config["target"]])
    y = df[config["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Train
    print("Training random forest...")
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"],
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1],
    }

    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1 Score:  {metrics['f1_score']}")

    # Check thresholds
    if metrics["accuracy"] < config["min_accuracy"]:
        print(
            f"\nWARNING: Accuracy {metrics['accuracy']} is below threshold {config['min_accuracy']}"
        )
    if metrics["f1_score"] < config["min_f1"]:
        print(
            f"\nWARNING: F1 {metrics['f1_score']} is below threshold {config['min_f1']}"
        )

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics_path = "metrics/results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics, config


if __name__ == "__main__":
    metrics, config = train_model()

    # Exit with error if thresholds not met
    if metrics["accuracy"] < config["min_accuracy"]:
        print(f"\nFAILED: Accuracy below threshold")
        sys.exit(1)
    if metrics["f1_score"] < config["min_f1"]:
        print(f"\nFAILED: F1 score below threshold")
        sys.exit(1)

    print("\nAll thresholds passed!")
