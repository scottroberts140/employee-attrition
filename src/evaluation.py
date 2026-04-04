from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def get_threshold_failures(
    metrics: dict, model_configs: dict
) -> dict[str, dict[str, float]]:
    """Return metrics that did not meet configured thresholds.

    Parameters
    ----------
    metrics : dict
        Computed metric values from ``evaluate_model``.
    model_configs : dict
        Run configuration containing requested metrics and thresholds.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of metric names to the observed value and configured threshold.

    Examples
    --------
    >>> get_threshold_failures({"accuracy": 0.7}, {"metrics": {"accuracy": 0.8}})
    """
    failures = {}

    for metric_name, threshold in model_configs.get("metrics", {}).items():
        if threshold is None or metric_name not in metrics:
            continue

        metric_value = float(metrics[metric_name])
        if metric_value < threshold:
            failures[metric_name] = {
                "value": metric_value,
                "threshold": float(threshold),
            }

    return failures


def evaluate_model(model, X_train, X_test, y_test, model_configs: dict):
    """Evaluate a trained model using the metrics requested in the run config.

    Parameters
    ----------
    model : object
        Trained model that supports ``predict`` and, when needed, ``predict_proba``.
    X_train : pandas.DataFrame
        Training feature matrix used to report metadata such as feature count and
        training set size.
    X_test : pandas.DataFrame
        Test feature matrix used for prediction.
    y_test : pandas.Series or array-like
        True target values for the test set.
    model_configs : dict
        Merged run configuration containing the ``metrics`` section that controls
        which evaluation metrics are computed and which thresholds are enforced.

    Returns
    -------
    dict
        Dictionary containing always-reported run metadata (`train_size`,
        `test_size`, and `n_features`) plus any requested evaluation metrics.

    Examples
    --------
    >>> metrics = evaluate_model(model, X_train, X_test, y_test, {"metrics": {"accuracy": 0.8, "f1": None}})
    >>> metrics["accuracy"]
    """
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

    threshold_failures = get_threshold_failures(metrics, model_configs)
    for metric_name, failure in threshold_failures.items():
        print(
            f"\nWARNING: {metric_name} {failure['value']:.4f} is below threshold {failure['threshold']}"
        )

    print(f"\nResults:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")

    return metrics
