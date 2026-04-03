import pandas as pd
import os
import sys
from typing import Type
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Add src to path so we can import preprocessing
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (
    validate_dataframe,
    clean_data,
    encode_categoricals,
    check_data_quality,
    encode_binary_column,
)

MODEL_TYPES = [
    "RF",
    "LR",
    "GB",
]


def load_data(url):
    """Load a dataset from a CSV file.

    Parameters
    ----------
    url : str
        Path or URL to the CSV file to load.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Examples
    --------
    >>> load_data("./data/raw/WA_Fn-UseC_-HR-Employee-Attrition.CSV")
    """
    print(f"Loading data from {url}...")
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def get_model_params(model_class: Type, config: dict) -> dict:
    """Return only the configuration keys accepted by a model class.

    Parameters
    ----------
    model_class : type
        Estimator class whose accepted parameter names should be used.
    config : dict
        Merged run configuration containing pipeline settings and model
        hyperparameters.

    Returns
    -------
    dict
        Subset of ``config`` containing only keys supported by the estimator.

    Examples
    --------
    >>> get_model_params(RandomForestClassifier, {"n_estimators": 100, "target": "Attrition"})
    """
    valid_params = model_class().get_params().keys()
    return {key: value for key, value in config.items() if key in valid_params}


def train_model(model_configs: dict):
    """Train a model using the merged experiment configuration.

    Parameters
    ----------
    model_configs : dict
        Merged run configuration containing dataset paths, preprocessing
        settings, split settings, target definition, and model hyperparameters.

    Returns
    -------
    tuple
        Tuple containing the trained model, ``X_train``, ``y_train``, ``X_test``,
        and ``y_test``.

    Examples
    --------
    >>> config = {"data_url_raw": "./data/raw/file.csv", "features_to_drop": [], "numeric_columns": [], "categorical_columns": [], "target": "Attrition", "test_size": 0.2, "random_state": 75, "model_type": "RF"}
    >>> train_model(config)
    """
    # Load
    df = load_data(model_configs["data_url_raw"])

    # Drop insignificant features
    df = df.drop(columns=model_configs["features_to_drop"])

    # Validate
    required = (
        model_configs["numeric_columns"]
        + model_configs["categorical_columns"]
        + [model_configs["target"]]
    )
    validate_dataframe(df, required, model_configs["target"])

    # Data quality check
    quality = check_data_quality(df, model_configs["numeric_columns"])
    print(
        f"Data quality: {quality['total_nulls']} nulls, {quality['duplicate_rows']} duplicates"
    )

    # Clean
    df = clean_data(
        df, model_configs["numeric_columns"], model_configs["categorical_columns"]
    )

    # Encode
    df = encode_categoricals(df, model_configs["categorical_columns"])

    # Encode target column
    df = encode_binary_column(df, model_configs["target"], "Yes")

    # Split
    X = df.drop(columns=[model_configs["target"]])
    y = df[model_configs["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=model_configs["test_size"],
        random_state=model_configs["random_state"],
        stratify=y,
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Train
    model_type = model_configs["model_type"]
    print(f"Training model type '{model_type}' ...")

    if model_type == "RF":
        model_params = get_model_params(RandomForestClassifier, model_configs)
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
    # elif model_type == "LR":
    #     model = None
    # elif model_type == "GB":
    #     model = None
    else:
        raise NotImplementedError(
            (f"Training for model type '{model_type}' not implemented")
        )

    return model, X_train, y_train, X_test, y_test
