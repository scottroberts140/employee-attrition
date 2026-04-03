import argparse
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import mlflow
from mlflow import sklearn as mlflow_sklearn
import yaml

# Add src to path so we can import preprocessing
sys.path.insert(0, os.path.dirname(__file__))
from train import (
    train_model,
    MODEL_TYPES,
)

from evaluation import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"
MODEL_CONFIG_DIR = PROJECT_ROOT / "configs" / "models"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"


def load_yaml(file_path: Path) -> dict:
    with file_path.open("r") as file:
        return yaml.safe_load(file)


def merge_dicts(base: dict, updates: dict) -> dict:
    merged = base.copy()

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def flatten_config_for_mlflow(config: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config structures into MLflow-friendly parameter values."""
    flattened = {}

    for key, value in config.items():
        flat_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_config_for_mlflow(value, prefix=flat_key))
        elif isinstance(value, list):
            flattened[flat_key] = json.dumps(value)
        elif value is None:
            flattened[flat_key] = "None"
        elif isinstance(value, (str, int, float, bool)):
            flattened[flat_key] = value
        else:
            flattened[flat_key] = str(value)

    return flattened


def get_dataset_config(dataset_name: str) -> dict:
    dataset_config_path = DATASET_CONFIG_DIR / f"{dataset_name}.yaml"
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")

    return load_yaml(dataset_config_path)


def get_experiment(exp: str):
    experiment_path = EXPERIMENT_DIR / f"{exp}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    return load_yaml(experiment_path)


def get_model_config_file(dataset_name: str, model_config_name: str) -> Path:
    model_config_path = MODEL_CONFIG_DIR / dataset_name / f"{model_config_name}.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    return model_config_path


def get_model_configurations(experiment: dict) -> list[dict]:
    exp_model_configs = []
    dataset_name = experiment["dataset_config"]
    dataset_config = get_dataset_config(dataset_name)

    experiment_definitions = experiment.get("experiments", {})
    if not experiment_definitions:
        raise ValueError("Experiment file must define at least one experiment")

    for experiment_name, experiment_config in experiment_definitions.items():
        experiment_defaults = experiment_config.get("defaults", {})
        experiment_title = experiment_config.get("title", experiment_name)
        model_entries = experiment_config.get("models", [])

        for model_entry in model_entries:
            model_configs_file = model_entry["file"]
            model_config_path = get_model_config_file(dataset_name, model_configs_file)
            model_configs = load_yaml(model_config_path)

            configurations = model_entry.get("configurations", [])
            include_all = len(configurations) == 0
            global_model_configs = model_configs.get("_global_", {})
            model_type = global_model_configs.get("model_type")

            if model_type not in MODEL_TYPES:
                raise ValueError(
                    f"Invalid model type '{model_type}' in file '{model_configs_file}'"
                )

            for mc_key, mc_value in model_configs.items():
                if mc_key == "_global_":
                    continue

                if include_all or mc_key in configurations:
                    exp_model_config = merge_dicts(dataset_config, experiment_defaults)
                    exp_model_config = merge_dicts(
                        exp_model_config, global_model_configs
                    )
                    exp_model_config = merge_dicts(exp_model_config, mc_value)
                    exp_model_config["dataset_name"] = dataset_name
                    exp_model_config["experiment_name"] = experiment_name
                    exp_model_config["experiment_title"] = experiment_title
                    exp_model_config["run_name"] = mc_value.get("title", mc_key)
                    exp_model_configs.append(exp_model_config)

    return exp_model_configs


def save_model(model, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(model, file)
    print(f"\nModel saved to {model_path}")


def save_metrics(metrics: dict, metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=2)
    print(f"Metrics saved to {metrics_path}")


def should_save_local_artifacts(model_config: dict) -> bool:
    """Return True when local model and metric files should be written."""
    return bool(model_config.get("save_local_artifacts", False))


def get_local_output_path(model_config: dict, artifact_type: str) -> Path:
    """Return the configured local output path or the default project path."""
    default_paths = {
        "model": PROJECT_ROOT / "models" / "model.pkl",
        "metrics": PROJECT_ROOT / "metrics" / "results.json",
    }
    configured_paths = model_config.get("local_output_paths", {})
    configured_path = configured_paths.get(artifact_type)

    if configured_path is None:
        return default_paths[artifact_type]

    return PROJECT_ROOT / configured_path


def run_experiment(exp: str):
    experiment = get_experiment(exp)
    exp_model_configs = get_model_configurations(experiment)

    current_dir = os.getcwd()
    tracking_uri = os.path.join(current_dir, "mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    for emc in exp_model_configs:
        mlflow.set_experiment(emc["experiment_title"])

        with mlflow.start_run(run_name=emc["run_name"]):
            mlflow.log_params(flatten_config_for_mlflow(emc))
            model, X_train, y_train, X_test, y_test = train_model(emc)
            metrics = evaluate_model(model, X_train, X_test, y_test, emc)
            mlflow.log_metrics(metrics)
            mlflow_sklearn.log_model(model, name="model")

            if should_save_local_artifacts(emc):
                save_model(model, get_local_output_path(emc, "model"))
                save_metrics(metrics, get_local_output_path(emc, "metrics"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment yaml filename without the .yaml extension",
    )
    args = parser.parse_args()

    run_experiment(args.exp)
