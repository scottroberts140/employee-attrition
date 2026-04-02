import json
import os
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(model, X_train, X_test, y_test, model_configs: dict):
    requested_metrics = model_configs.get("metrics", {})
    metric_functions = {
        "accuracy": lambda y_true, y_pred, y_prob: float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": lambda y_true, y_pred, y_prob: float(
            precision_score(y_true, y_pred)
        ),
        "recall": lambda y_true, y_pred, y_prob: float(recall_score(y_true, y_pred)),
        "f1": lambda y_true, y_pred, y_prob: float(f1_score(y_true, y_pred)),
        "auc_roc": lambda y_true, y_pred, y_prob: float(roc_auc_score(y_true, y_prob)),
    }

    y_pred = model.predict(X_test)

    y_prob = None
    if "auc_roc" in requested_metrics:
        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "Model does not support predict_proba, which is required for auc_roc"
            )
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1],
    }

    for metric_name, threshold in requested_metrics.items():
        if metric_name not in metric_functions:
            raise ValueError(f"Unknown metric: {metric_name}")

        metric_value = metric_functions[metric_name](y_test, y_pred, y_prob)
        metrics[metric_name] = round(metric_value, 4)

        if threshold is not None and metric_value < threshold:
            print(
                f"\nWARNING: {metric_name} {metrics[metric_name]} is below threshold {threshold}"
            )

    print(f"\nResults:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")

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
