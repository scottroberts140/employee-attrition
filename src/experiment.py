import os
import sys
import yaml
import argparse

# Add src to path so we can import preprocessing
sys.path.insert(0, os.path.dirname(__file__))
from train import (
    train_model,
    MODEL_TYPES,
)

from evaluation import evaluate_model


def get_default_config():
    with open("./configs/config_.yaml", "r") as file:
        return yaml.safe_load(file)


def get_experiment(exp: str):
    with open(f"./experiment/{exp}.yaml", "r") as file:
        return yaml.safe_load(file)


def get_model_configurations(experiment: dict, default_configs: dict) -> list[dict]:
    exp_model_configs = []

    for exp in experiment.values():
        model_configs_file = exp["file"]

        with open(f"./configs/{model_configs_file}.yaml", "r") as file:
            model_configs = yaml.safe_load(file)

        configurations = exp["configurations"]
        include_all = len(configurations) == 0
        global_model_configs = model_configs["_global_"]
        model_type = global_model_configs.get("model_type")

        if model_type not in MODEL_TYPES:
            raise ValueError(
                (f"Invalid model type '{model_type}' in file '{model_configs_file}")
            )

        for mc_key, mc_value in model_configs.items():
            if mc_key == "_global_":
                continue

            if include_all or mc_key in configurations:
                exp_model_config = default_configs.copy()
                exp_model_config.update(global_model_configs)
                exp_model_config.update(mc_value)
                exp_model_configs.append(exp_model_config)

    return exp_model_configs


def run_experiment(exp: str):
    config = get_default_config()
    experiment = get_experiment(exp)
    exp_model_configs = get_model_configurations(experiment, config)

    for emc in exp_model_configs:
        model, X_train, y_train, X_test, y_test = train_model(emc)
        evaluate_model(model, X_train, X_test, y_test, emc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment yaml filename without the .yaml extension",
    )
    args = parser.parse_args()

    run_experiment(args.exp)
