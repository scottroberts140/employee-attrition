from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "src")

import experiment
from evaluation import evaluate_model
import train
from train import get_model_params


class PredictOnlyModel:
    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        return self._predictions


class PredictProbaModel(PredictOnlyModel):
    def __init__(self, predictions, probabilities):
        super().__init__(predictions)
        self._probabilities = np.array(probabilities)

    def predict_proba(self, X):
        return self._probabilities


def test_merge_dicts_recursively_merges_nested_values():
    base = {"metrics": {"accuracy": 0.5}, "model_type": "RF"}
    updates = {"metrics": {"f1": 0.3}, "n_estimators": 100}

    result = experiment.merge_dicts(base, updates)

    assert result == {
        "metrics": {"accuracy": 0.5, "f1": 0.3},
        "model_type": "RF",
        "n_estimators": 100,
    }


def test_flatten_config_for_mlflow_flattens_nested_values():
    config = {
        "metrics": {"accuracy": 0.5, "f1": None},
        "tags": ["baseline", "rf"],
        "save_local_artifacts": False,
    }

    result = experiment.flatten_config_for_mlflow(config)

    assert result["metrics.accuracy"] == 0.5
    assert result["metrics.f1"] == "None"
    assert result["tags"] == '["baseline", "rf"]'
    assert result["save_local_artifacts"] is False


def test_get_model_configurations_merges_dataset_experiment_and_model_configs(
    tmp_path, monkeypatch
):
    dataset_dir = tmp_path / "configs" / "datasets"
    model_dir = tmp_path / "configs" / "models" / "toy_dataset"
    experiment_dir = tmp_path / "experiments"
    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)

    (dataset_dir / "toy_dataset.yaml").write_text(
        """
target: "Attrition"
data_url_raw: "./toy.csv"
numeric_columns: ["Age"]
categorical_columns: ["Dept"]
features_to_drop: []
test_size: 0.2
random_state: 75
""".strip()
    )

    (model_dir / "rf.yaml").write_text(
        """
_global_:
  model_type: "RF"
  n_estimators: 25
  metrics:
    f1: 0.3

baseline:
  title: "Toy RF Baseline"
  metrics:
    precision:
""".strip()
    )

    monkeypatch.setattr(experiment, "DATASET_CONFIG_DIR", dataset_dir)
    monkeypatch.setattr(experiment, "MODEL_CONFIG_DIR", tmp_path / "configs" / "models")
    monkeypatch.setattr(experiment, "EXPERIMENT_DIR", experiment_dir)

    experiment_config = {
        "dataset_config": "toy_dataset",
        "experiments": {
            "baseline": {
                "title": "Toy Experiment",
                "defaults": {"metrics": {"accuracy": 0.6}},
                "models": [{"file": "rf", "configurations": []}],
            }
        },
    }

    results = experiment.get_model_configurations(experiment_config)

    assert len(results) == 1
    result = results[0]
    assert result["dataset_name"] == "toy_dataset"
    assert result["experiment_name"] == "baseline"
    assert result["experiment_title"] == "Toy Experiment"
    assert result["run_name"] == "Toy RF Baseline"
    assert result["model_type"] == "RF"
    assert result["n_estimators"] == 25
    assert result["metrics"] == {
        "accuracy": 0.6,
        "f1": 0.3,
        "precision": None,
    }


def test_should_save_local_artifacts_defaults_false_and_respects_flag():
    assert experiment.should_save_local_artifacts({}) is False
    assert (
        experiment.should_save_local_artifacts({"save_local_artifacts": True}) is True
    )


def test_get_local_output_path_uses_default_and_custom_paths():
    default_model_path = experiment.get_local_output_path({}, "model")
    custom_metrics_path = experiment.get_local_output_path(
        {"local_output_paths": {"metrics": "metrics/custom/results.json"}},
        "metrics",
    )

    assert default_model_path == experiment.PROJECT_ROOT / "models" / "model.pkl"
    assert (
        custom_metrics_path == experiment.PROJECT_ROOT / "metrics/custom/results.json"
    )


def test_get_data_version_info_returns_dvc_md5_when_sidecar_exists(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    dataset_path = data_dir / "toy.csv"
    dataset_path.write_text("x\n1\n")
    (data_dir / "toy.csv.dvc").write_text(
        """
outs:
  - md5: abc123
    size: 4
    hash: md5
    path: toy.csv
""".strip()
    )

    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)

    result = experiment.get_data_version_info({"data_url_raw": "./data/raw/toy.csv"})

    assert result == {
        "data_version": "abc123",
        "data_version_source": "dvc_md5",
    }


def test_get_data_version_info_falls_back_to_file_name_when_no_dvc(
    monkeypatch, tmp_path
):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "toy.csv").write_text("x\n1\n")

    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)

    result = experiment.get_data_version_info({"data_url_raw": "./data/raw/toy.csv"})

    assert result == {
        "data_version": "toy.csv",
        "data_version_source": "file_name",
    }


def test_get_model_params_filters_to_supported_estimator_keys():
    config = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 75,
        "target": "Attrition",
        "metrics": {"accuracy": 0.5},
    }

    result = get_model_params(RandomForestClassifier, config)

    assert result == {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 75,
    }


def test_evaluate_model_returns_requested_metrics_and_metadata():
    model = PredictProbaModel(
        predictions=[0, 1],
        probabilities=[[0.9, 0.1], [0.1, 0.9]],
    )
    X_train = pd.DataFrame({"x1": [0, 1, 0]})
    X_test = pd.DataFrame({"x1": [0, 1]})
    y_test = pd.Series([0, 1])

    result = evaluate_model(
        model,
        X_train,
        X_test,
        y_test,
        {"metrics": {"accuracy": 0.5, "f1": None, "auc_roc": 0.5}},
    )

    assert result == {
        "train_size": 3,
        "test_size": 2,
        "n_features": 1,
        "accuracy": 1.0,
        "f1": 1.0,
        "auc_roc": 1.0,
    }


def test_evaluate_model_warns_only_for_metrics_with_thresholds(capsys):
    model = PredictProbaModel(
        predictions=[0, 1, 0, 0],
        probabilities=[[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.9, 0.1]],
    )
    X_train = pd.DataFrame({"x1": [0, 1, 0, 1]})
    X_test = pd.DataFrame({"x1": [0, 1, 0, 1]})
    y_test = pd.Series([0, 1, 1, 0])

    evaluate_model(
        model,
        X_train,
        X_test,
        y_test,
        {"metrics": {"f1": 0.8, "precision": None}},
    )

    output = capsys.readouterr().out
    assert "WARNING: f1" in output
    assert "WARNING: precision" not in output


def test_evaluate_model_raises_for_unknown_metric():
    model = PredictOnlyModel(predictions=[0, 1])
    X_train = pd.DataFrame({"x1": [0, 1]})
    X_test = pd.DataFrame({"x1": [0, 1]})
    y_test = pd.Series([0, 1])

    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate_model(model, X_train, X_test, y_test, {"metrics": {"bogus": 1.0}})


def test_evaluate_model_raises_when_auc_requested_without_predict_proba():
    model = PredictOnlyModel(predictions=[0, 1])
    X_train = pd.DataFrame({"x1": [0, 1]})
    X_test = pd.DataFrame({"x1": [0, 1]})
    y_test = pd.Series([0, 1])

    with pytest.raises(ValueError, match="predict_proba"):
        evaluate_model(model, X_train, X_test, y_test, {"metrics": {"auc_roc": 0.5}})


def test_train_model_trains_random_forest_on_synthetic_data(monkeypatch):
    df = pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45, 50],
            "Dept": ["Sales", "HR", "Sales", "HR", "Sales", "HR"],
            "Attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )

    monkeypatch.setattr(train, "load_data", lambda url: df)

    config = {
        "data_url_raw": "./unused.csv",
        "features_to_drop": [],
        "numeric_columns": ["Age"],
        "categorical_columns": ["Dept"],
        "target": "Attrition",
        "test_size": 0.5,
        "random_state": 0,
        "model_type": "RF",
        "n_estimators": 5,
        "max_depth": 2,
    }

    model, X_train, y_train, X_test, y_test = train.train_model(config)

    assert isinstance(model, RandomForestClassifier)
    assert len(X_train) == 3
    assert len(X_test) == 3
    assert len(y_train) == 3
    assert len(y_test) == 3
    assert "Attrition" not in X_train.columns
    assert X_train.shape[1] >= 2


def test_train_model_trains_logistic_regression_on_synthetic_data(monkeypatch):
    df = pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45, 50],
            "Dept": ["Sales", "HR", "Sales", "HR", "Sales", "HR"],
            "Attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )

    monkeypatch.setattr(train, "load_data", lambda url: df)

    config = {
        "data_url_raw": "./unused.csv",
        "features_to_drop": [],
        "numeric_columns": ["Age"],
        "categorical_columns": ["Dept"],
        "target": "Attrition",
        "test_size": 0.5,
        "random_state": 0,
        "model_type": "LR",
        "solver": "liblinear",
        "max_iter": 200,
        "C": 1.0,
    }

    model, X_train, y_train, X_test, y_test = train.train_model(config)

    assert isinstance(model, LogisticRegression)
    assert len(X_train) == 3
    assert len(X_test) == 3
    assert "Attrition" not in X_train.columns


def test_train_model_trains_gradient_boosting_with_additional_feature_drop(monkeypatch):
    df = pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45, 50],
            "MonthlyIncome": [3000, 3200, 3400, 3600, 3800, 4000],
            "Dept": ["Sales", "HR", "Sales", "HR", "Sales", "HR"],
            "Attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )

    monkeypatch.setattr(train, "load_data", lambda url: df)

    config = {
        "data_url_raw": "./unused.csv",
        "features_to_drop": [],
        "additional_features_to_drop": ["MonthlyIncome"],
        "numeric_columns": ["Age", "MonthlyIncome"],
        "categorical_columns": ["Dept"],
        "target": "Attrition",
        "test_size": 0.5,
        "random_state": 0,
        "model_type": "GB",
        "learning_rate": 0.1,
        "n_estimators": 10,
        "max_depth": 2,
    }

    model, X_train, y_train, X_test, y_test = train.train_model(config)

    assert isinstance(model, GradientBoostingClassifier)
    assert len(X_train) == 3
    assert len(X_test) == 3
    assert "MonthlyIncome" not in X_train.columns
    assert "Attrition" not in X_train.columns


def test_run_experiment_logs_to_mlflow_and_skips_local_saves_when_disabled(
    monkeypatch,
):
    run_config = {
        "experiment_title": "Toy Experiment",
        "run_name": "baseline",
        "save_local_artifacts": False,
        "metrics": {"accuracy": 0.5},
        "model_type": "RF",
    }
    returned_metrics = {
        "train_size": 3,
        "test_size": 2,
        "n_features": 1,
        "accuracy": 1.0,
    }
    dummy_model = object()
    X_train = pd.DataFrame({"x1": [0, 1, 0]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"x1": [0, 1]})
    y_test = pd.Series([0, 1])

    calls = {
        "tracking_uri": None,
        "experiment": [],
        "params": [],
        "metrics": [],
        "models": [],
        "saved_model": 0,
        "saved_metrics": 0,
        "run_name": None,
    }

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(experiment, "get_experiment", lambda exp: {"stub": True})
    monkeypatch.setattr(
        experiment, "get_model_configurations", lambda exp: [run_config]
    )
    monkeypatch.setattr(
        experiment,
        "train_model",
        lambda config: (dummy_model, X_train, y_train, X_test, y_test),
    )
    monkeypatch.setattr(
        experiment,
        "evaluate_model",
        lambda model, Xtr, Xte, yte, config: returned_metrics,
    )
    monkeypatch.setattr(
        experiment,
        "get_data_version_info",
        lambda config: {"data_version": "abc123", "data_version_source": "dvc_md5"},
    )
    monkeypatch.setattr(
        experiment.mlflow,
        "set_tracking_uri",
        lambda uri: calls.__setitem__("tracking_uri", uri),
    )
    monkeypatch.setattr(
        experiment.mlflow, "get_tracking_uri", lambda: "file:///tmp/mlruns"
    )
    monkeypatch.setattr(
        experiment.mlflow,
        "set_experiment",
        lambda name: calls["experiment"].append(name),
    )
    monkeypatch.setattr(
        experiment.mlflow,
        "log_params",
        lambda params: calls["params"].append(params),
    )
    monkeypatch.setattr(
        experiment.mlflow,
        "log_metrics",
        lambda metrics: calls["metrics"].append(metrics),
    )
    monkeypatch.setattr(
        experiment.mlflow,
        "start_run",
        lambda run_name=None: calls.__setitem__("run_name", run_name) or DummyRun(),
    )
    monkeypatch.setattr(
        experiment.mlflow_sklearn,
        "log_model",
        lambda model, name: calls["models"].append((model, name)),
    )
    monkeypatch.setattr(
        experiment,
        "save_model",
        lambda model, path: calls.__setitem__("saved_model", calls["saved_model"] + 1),
    )
    monkeypatch.setattr(
        experiment,
        "save_metrics",
        lambda metrics, path: calls.__setitem__(
            "saved_metrics", calls["saved_metrics"] + 1
        ),
    )

    experiment.run_experiment("experiment_baseline")

    assert calls["tracking_uri"] == f"file://{Path.cwd() / 'mlruns'}"
    assert calls["experiment"] == ["Toy Experiment"]
    assert calls["run_name"] == "baseline"
    expected_params = experiment.flatten_config_for_mlflow(run_config)
    expected_params.update({"data_version": "abc123", "data_version_source": "dvc_md5"})
    assert calls["params"] == [expected_params]
    assert calls["metrics"] == [returned_metrics]
    assert calls["models"] == [(dummy_model, "model")]
    assert calls["saved_model"] == 0
    assert calls["saved_metrics"] == 0


def test_run_experiment_saves_local_artifacts_with_custom_paths(monkeypatch):
    run_config = {
        "experiment_title": "Toy Experiment",
        "run_name": "baseline",
        "save_local_artifacts": True,
        "local_output_paths": {
            "model": "models/custom/toy_model.pkl",
            "metrics": "metrics/custom/toy_metrics.json",
        },
        "metrics": {"accuracy": 0.5},
        "model_type": "RF",
    }
    returned_metrics = {
        "train_size": 3,
        "test_size": 2,
        "n_features": 1,
        "accuracy": 1.0,
    }
    dummy_model = object()
    X_train = pd.DataFrame({"x1": [0, 1, 0]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"x1": [0, 1]})
    y_test = pd.Series([0, 1])

    saved_paths = {"model": None, "metrics": None}

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(experiment, "get_experiment", lambda exp: {"stub": True})
    monkeypatch.setattr(
        experiment, "get_model_configurations", lambda exp: [run_config]
    )
    monkeypatch.setattr(
        experiment,
        "train_model",
        lambda config: (dummy_model, X_train, y_train, X_test, y_test),
    )
    monkeypatch.setattr(
        experiment,
        "evaluate_model",
        lambda model, Xtr, Xte, yte, config: returned_metrics,
    )
    monkeypatch.setattr(
        experiment,
        "get_data_version_info",
        lambda config: {"data_version": "abc123", "data_version_source": "dvc_md5"},
    )
    monkeypatch.setattr(experiment.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        experiment.mlflow, "get_tracking_uri", lambda: "file:///tmp/mlruns"
    )
    monkeypatch.setattr(experiment.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(experiment.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(experiment.mlflow, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(
        experiment.mlflow, "start_run", lambda run_name=None: DummyRun()
    )
    monkeypatch.setattr(
        experiment.mlflow_sklearn, "log_model", lambda model, name: None
    )
    monkeypatch.setattr(
        experiment,
        "save_model",
        lambda model, path: saved_paths.__setitem__("model", path),
    )
    monkeypatch.setattr(
        experiment,
        "save_metrics",
        lambda metrics, path: saved_paths.__setitem__("metrics", path),
    )

    experiment.run_experiment("experiment_baseline")

    assert (
        saved_paths["model"] == experiment.PROJECT_ROOT / "models/custom/toy_model.pkl"
    )
    assert (
        saved_paths["metrics"]
        == experiment.PROJECT_ROOT / "metrics/custom/toy_metrics.json"
    )
