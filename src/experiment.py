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
    """Load a YAML file into a dictionary.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the YAML file to load.

    Returns
    -------
    dict
        Parsed YAML contents.

    Examples
    --------
    >>> load_yaml(Path("configs/datasets/employee_attrition.yaml"))
    """
    with file_path.open("r") as file:
        return yaml.safe_load(file)


def merge_dicts(base: dict, updates: dict) -> dict:
    """Recursively merge one dictionary into another.

    Parameters
    ----------
    base : dict
        Base dictionary whose values are used as defaults.
    updates : dict
        Dictionary containing overriding values.

    Returns
    -------
    dict
        A new dictionary containing the merged values.

    Examples
    --------
    >>> merge_dicts({"metrics": {"accuracy": 0.5}}, {"metrics": {"f1": 0.3}})
    """
    merged = base.copy()

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def flatten_config_for_mlflow(config: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config structures into MLflow-friendly parameter values.

    Parameters
    ----------
    config : dict
        Configuration dictionary to flatten.
    prefix : str, default=""
        Prefix used during recursive flattening.

    Returns
    -------
    dict[str, Any]
        Flat dictionary suitable for ``mlflow.log_params``.

    Examples
    --------
    >>> flatten_config_for_mlflow({"metrics": {"accuracy": 0.5}, "tags": ["baseline"]})
    """
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


def get_data_version_info(model_config: dict) -> dict[str, str]:
    """Return data-version metadata for the configured dataset.

    Parameters
    ----------
    model_config : dict
        Merged run configuration containing the dataset path.

    Returns
    -------
    dict[str, str]
        Dictionary containing the data version value and its source.

    Examples
    --------
    >>> get_data_version_info({"data_url_raw": "./data/raw/example.csv"})
    """
    data_path = PROJECT_ROOT / model_config["data_url_raw"]
    data_path = data_path.resolve()
    dvc_path = Path(f"{data_path}.dvc")

    if dvc_path.exists():
        dvc_info = load_yaml(dvc_path)
        for out in dvc_info.get("outs", []):
            if out.get("path") == data_path.name and out.get("md5"):
                return {
                    "data_version": out["md5"],
                    "data_version_source": "dvc_md5",
                }

    return {
        "data_version": data_path.name,
        "data_version_source": "file_name",
    }


def get_dataset_config(dataset_name: str) -> dict:
    """Load the shared dataset configuration for an experiment.

    Parameters
    ----------
    dataset_name : str
        Dataset configuration name without the ``.yaml`` extension.

    Returns
    -------
    dict
        Dataset configuration contents.

    Examples
    --------
    >>> get_dataset_config("employee_attrition")
    """
    dataset_config_path = DATASET_CONFIG_DIR / f"{dataset_name}.yaml"
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")

    return load_yaml(dataset_config_path)


def get_experiment(exp: str):
    """Load an experiment definition file.

    Parameters
    ----------
    exp : str
        Experiment filename without the ``.yaml`` extension.

    Returns
    -------
    dict
        Experiment configuration contents.

    Examples
    --------
    >>> get_experiment("experiment_baseline")
    """
    experiment_path = EXPERIMENT_DIR / f"{exp}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    return load_yaml(experiment_path)


def get_model_config_file(dataset_name: str, model_config_name: str) -> Path:
    """Return the path to a dataset-specific model configuration file.

    Parameters
    ----------
    dataset_name : str
        Dataset configuration name used to locate the model subfolder.
    model_config_name : str
        Model configuration filename without the ``.yaml`` extension.

    Returns
    -------
    pathlib.Path
        Path to the model configuration file.

    Examples
    --------
    >>> get_model_config_file("employee_attrition", "rf")
    """
    model_config_path = MODEL_CONFIG_DIR / dataset_name / f"{model_config_name}.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    return model_config_path


def get_model_configurations(experiment: dict) -> list[dict]:
    """Build merged model configurations for every run in an experiment file.

    Parameters
    ----------
    experiment : dict
        Experiment definition containing dataset, experiment, and model references.

    Returns
    -------
    list[dict]
        List of merged run configurations ready for training.

    Examples
    --------
    >>> experiment = get_experiment("experiment_baseline")
    >>> get_model_configurations(experiment)
    """
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
    """Save a trained model to a local pickle file.

    Parameters
    ----------
    model : object
        Trained model object to serialize.
    model_path : pathlib.Path
        Output path for the pickle file.

    Returns
    -------
    None
        This function writes the model to disk.

    Examples
    --------
    >>> save_model(model, PROJECT_ROOT / "models" / "model.pkl")
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(model, file)
    print(f"\nModel saved to {model_path}")


def save_metrics(metrics: dict, metrics_path: Path) -> None:
    """Save evaluation metrics to a local JSON file.

    Parameters
    ----------
    metrics : dict
        Metric names and values to write.
    metrics_path : pathlib.Path
        Output path for the JSON file.

    Returns
    -------
    None
        This function writes the metrics to disk.

    Examples
    --------
    >>> save_metrics({"accuracy": 0.9}, PROJECT_ROOT / "metrics" / "results.json")
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=2)
    print(f"Metrics saved to {metrics_path}")


def should_save_local_artifacts(model_config: dict) -> bool:
    """Return whether local model and metric files should be written.

    Parameters
    ----------
    model_config : dict
        Merged run configuration.

    Returns
    -------
    bool
        ``True`` when local artifacts should be saved.

    Examples
    --------
    >>> should_save_local_artifacts({"save_local_artifacts": True})
    """
    return bool(model_config.get("save_local_artifacts", False))


def get_local_output_path(model_config: dict, artifact_type: str) -> Path:
    """Return the configured local output path or the default project path.

    Parameters
    ----------
    model_config : dict
        Merged run configuration.
    artifact_type : str
        Artifact type to resolve. Supported values are ``model`` and ``metrics``.

    Returns
    -------
    pathlib.Path
        Resolved local output path.

    Examples
    --------
    >>> get_local_output_path({"local_output_paths": {"model": "models/custom.pkl"}}, "model")
    """
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
    """Run every configured model for an experiment definition.

    Parameters
    ----------
    exp : str
        Experiment filename without the ``.yaml`` extension.

    Returns
    -------
    None
        This function orchestrates training, evaluation, logging, and optional
        local artifact saving.

    Examples
    --------
    >>> run_experiment("experiment_baseline")
    """
    experiment = get_experiment(exp)
    exp_model_configs = get_model_configurations(experiment)

    current_dir = os.getcwd()
    tracking_uri = os.path.join(current_dir, "mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    for emc in exp_model_configs:
        mlflow.set_experiment(emc["experiment_title"])

        with mlflow.start_run(run_name=emc["run_name"]):
            run_params = flatten_config_for_mlflow(emc)
            run_params.update(get_data_version_info(emc))
            mlflow.log_params(run_params)
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
