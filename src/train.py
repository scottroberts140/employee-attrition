import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
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

# Configuration
CONFIG = {
    "data_url": "https://raw.githubusercontent.com/TripleTen-DS/Dataset/refs/heads/main/student_dropout_dataset.csv",
    "target": "Dropout",
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 10,
    "numeric_columns": [
        "Age",
        "Family_Income",
        "Study_Hours_per_Day",
        "Attendance_Rate",
        "Assignment_Delay_Days",
        "Travel_Time_Minutes",
        "Stress_Index",
        "GPA",
        "Semester_GPA",
        "CGPA",
    ],
    "categorical_columns": [
        "Gender",
        "Internet_Access",
        "Part_Time_Job",
        "Scholarship",
        "Semester",
        "Department",
        "Parental_Education",
    ],
    "min_accuracy": 0.35,
    "min_f1": 0.30,
}


def load_data(url):
    """Load dataset from URL."""
    print(f"Loading data from {url}...")
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def train_model(config=None):
    """Full training pipeline. Returns metrics dictionary."""
    if config is None:
        config = CONFIG

    # Load
    df = load_data(config["data_url"])

    # Drop ID column
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])

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

    return metrics


if __name__ == "__main__":
    metrics = train_model()

    # Exit with error if thresholds not met
    if metrics["accuracy"] < CONFIG["min_accuracy"]:
        print(f"\nFAILED: Accuracy below threshold")
        sys.exit(1)
    if metrics["f1_score"] < CONFIG["min_f1"]:
        print(f"\nFAILED: F1 score below threshold")
        sys.exit(1)

    print("\nAll thresholds passed!")
