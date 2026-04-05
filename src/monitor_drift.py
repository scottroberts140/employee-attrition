import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "drift_report.html"


def load_yaml(file_path: Path) -> dict:
    """Load a YAML file into a dictionary.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    with file_path.open("r") as file:
        return yaml.safe_load(file)


def load_dataset_config(dataset_name: str) -> dict:
    """Load the dataset configuration used by the training pipeline.

    Parameters
    ----------
    dataset_name : str
        Dataset config name without the ``.yaml`` extension.

    Returns
    -------
    dict
        Dataset configuration dictionary.
    """
    config_path = DATASET_CONFIG_DIR / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    return load_yaml(config_path)


def get_feature_columns(dataset_config: dict) -> list[str]:
    """Return the model input feature columns defined in the dataset config.

    Parameters
    ----------
    dataset_config : dict
        Dataset configuration.

    Returns
    -------
    list[str]
        Ordered list of active feature columns.
    """
    dropped_columns = set(dataset_config.get("features_to_drop", []))
    feature_columns = (
        dataset_config["numeric_columns"] + dataset_config["categorical_columns"]
    )
    return [column for column in feature_columns if column not in dropped_columns]


def prepare_feature_dataframe(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Select monitoring features and fill missing values for drift analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    feature_columns : list[str]
        Feature columns to keep.

    Returns
    -------
    pandas.DataFrame
        Monitoring dataframe with missing values filled.
    """
    prepared_df = df[feature_columns].copy()

    numeric_columns = prepared_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        prepared_df[column] = prepared_df[column].fillna(prepared_df[column].median())

    categorical_columns = prepared_df.columns.difference(numeric_columns)
    for column in categorical_columns:
        mode = prepared_df[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        prepared_df[column] = prepared_df[column].fillna(fill_value)

    return prepared_df


def split_reference_and_production(
    df: pd.DataFrame,
    target_column: str,
    production_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into a reference sample and a production sample.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataset containing features and target.
    target_column : str
        Target column name used for stratification.
    production_fraction : float
        Fraction of rows reserved for simulated production data.
    random_state : int
        Random seed used by ``train_test_split``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Reference dataset followed by production dataset.
    """
    reference_df, production_df = train_test_split(
        df,
        test_size=production_fraction,
        random_state=random_state,
        stratify=df[target_column],
    )
    return reference_df.reset_index(drop=True), production_df.reset_index(drop=True)


def introduce_synthetic_drift(
    production_df: pd.DataFrame, random_state: int = 42
) -> pd.DataFrame:
    """Introduce synthetic drift into a production dataset.

    Parameters
    ----------
    production_df : pandas.DataFrame
        Production dataframe before drift injection.
    random_state : int, default=42
        Seed used for reproducible drift generation.

    Returns
    -------
    pandas.DataFrame
        Drifted production dataframe.
    """
    drifted_df = production_df.copy()
    random_generator = np.random.default_rng(random_state)

    if "MonthlyIncome" in drifted_df.columns:
        multiplier = random_generator.uniform(1.15, 1.45, len(drifted_df))
        drifted_df["MonthlyIncome"] = drifted_df["MonthlyIncome"] * multiplier

    if "DistanceFromHome" in drifted_df.columns:
        distance_shift = random_generator.normal(4.0, 1.5, len(drifted_df))
        drifted_df["DistanceFromHome"] = np.clip(
            drifted_df["DistanceFromHome"] + distance_shift,
            1,
            None,
        )

    if "Age" in drifted_df.columns and len(drifted_df) > 0:
        affected_count = max(1, int(len(drifted_df) * 0.2))
        affected_indices = random_generator.choice(
            drifted_df.index,
            size=affected_count,
            replace=False,
        )
        drifted_df.loc[affected_indices, "Age"] = random_generator.integers(
            45,
            61,
            size=affected_count,
        )

    if "OverTime" in drifted_df.columns and len(drifted_df) > 0:
        overtime_count = max(1, int(len(drifted_df) * 0.3))
        overtime_indices = random_generator.choice(
            drifted_df.index,
            size=overtime_count,
            replace=False,
        )
        drifted_df.loc[overtime_indices, "OverTime"] = "Yes"

    if "Department" in drifted_df.columns and len(drifted_df) > 0:
        department_count = max(1, int(len(drifted_df) * 0.2))
        department_indices = random_generator.choice(
            drifted_df.index,
            size=department_count,
            replace=False,
        )
        drifted_df.loc[department_indices, "Department"] = "Sales"

    return drifted_df


def build_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    feature_columns: list[str],
    drift_detection_threshold: float,
) -> object:
    """Run Evidently drift detection across all configured features.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        Reference dataset.
    production_df : pandas.DataFrame
        Drifted production dataset.
    feature_columns : list[str]
        Feature columns to include.
    drift_detection_threshold : float
        Per-feature statistical threshold used by Evidently.

    Returns
    -------
    object
        Evidently snapshot containing metrics and HTML rendering methods.
    """
    from evidently import Dataset, Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset

    report = Report(
        [
            DataDriftPreset(
                columns=feature_columns,
                threshold=drift_detection_threshold,
            ),
            *[
                ValueDrift(column=column, threshold=drift_detection_threshold)
                for column in feature_columns
            ],
        ]
    )
    return report.run(
        Dataset.from_pandas(reference_df[feature_columns]),
        Dataset.from_pandas(production_df[feature_columns]),
    )


def summarize_drift(snapshot_dict: dict, feature_columns: list[str]) -> dict:
    """Extract per-feature and overall drift summary from an Evidently snapshot.

    Parameters
    ----------
    snapshot_dict : dict
        Serialized Evidently snapshot.
    feature_columns : list[str]
        Feature columns included in the report.

    Returns
    -------
    dict
        Summary containing drifted columns, count, and share.
    """
    drifted_columns = []

    for metric in snapshot_dict.get("metrics", []):
        config = metric.get("config", {})
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue

        metric_value = float(metric["value"])
        threshold = float(config.get("threshold", 0.05))
        if metric_value <= threshold:
            drifted_columns.append(
                {
                    "column": config["column"],
                    "method": config.get("method"),
                    "score": metric_value,
                    "threshold": threshold,
                }
            )

    drifted_columns.sort(key=lambda item: item["score"])
    drift_count = len(drifted_columns)
    total_columns = len(feature_columns)
    drift_share = drift_count / total_columns if total_columns else 0.0

    return {
        "drifted_columns": drifted_columns,
        "drift_count": drift_count,
        "total_columns": total_columns,
        "drift_share": drift_share,
    }


def print_drift_summary(summary: dict, report_path: Path) -> None:
    """Print the drift monitoring summary.

    Parameters
    ----------
    summary : dict
        Drift summary from ``summarize_drift``.
    report_path : pathlib.Path
        Output path for the saved HTML report.

    Returns
    -------
    None
        This function prints the summary to stdout.
    """
    print("Drift monitoring summary")
    print("=" * 80)
    print(
        f"Drifted features: {summary['drift_count']}/{summary['total_columns']} "
        f"({summary['drift_share']:.2%})"
    )

    if summary["drifted_columns"]:
        print("\nDrifted features:")
        for item in summary["drifted_columns"]:
            print(
                f"- {item['column']}: score={item['score']:.4f}, "
                f"threshold={item['threshold']:.4f}"
            )
    else:
        print("\nNo drifted features detected.")

    print(f"\nHTML report saved to: {report_path}")


def monitor_drift(
    dataset_name: str = "employee_attrition",
    report_path: Path = DEFAULT_REPORT_PATH,
    max_drift_share: float = 0.15,
    drift_detection_threshold: float = 0.05,
) -> int:
    """Run drift monitoring and return an appropriate process exit code.

    Parameters
    ----------
    dataset_name : str, default="employee_attrition"
        Dataset config name.
    report_path : pathlib.Path, default=DEFAULT_REPORT_PATH
        Output HTML report path.
    max_drift_share : float, default=0.15
        Maximum acceptable share of drifted features before failing.
    drift_detection_threshold : float, default=0.05
        Per-feature statistical threshold used by Evidently.

    Returns
    -------
    int
        ``0`` when drift is within bounds, otherwise ``1``.
    """
    dataset_config = load_dataset_config(dataset_name)
    dataset_path = PROJECT_ROOT / dataset_config["data_url_raw"]
    raw_df = pd.read_csv(dataset_path)

    reference_df, production_df = split_reference_and_production(
        raw_df,
        target_column=dataset_config["target"],
        production_fraction=dataset_config["test_size"],
        random_state=dataset_config["random_state"],
    )

    feature_columns = get_feature_columns(dataset_config)
    reference_features = prepare_feature_dataframe(reference_df, feature_columns)
    production_features = prepare_feature_dataframe(production_df, feature_columns)
    production_features = introduce_synthetic_drift(
        production_features,
        random_state=dataset_config["random_state"],
    )

    snapshot = build_drift_report(
        reference_features,
        production_features,
        feature_columns,
        drift_detection_threshold=drift_detection_threshold,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))

    summary = summarize_drift(snapshot.dict(), feature_columns)
    print_drift_summary(summary, report_path)

    if summary["drift_share"] > max_drift_share:
        print(
            f"\nDrift share {summary['drift_share']:.2%} exceeds "
            f"threshold {max_drift_share:.2%}."
        )
        return 1

    print(
        f"\nDrift share {summary['drift_share']:.2%} is within threshold "
        f"{max_drift_share:.2%}."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        default="employee_attrition",
        help="Dataset config name without the .yaml extension",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Output path for the Evidently HTML report",
    )
    parser.add_argument(
        "--max-drift-share",
        type=float,
        default=0.15,
        help="Exit with code 1 when the share of drifted features exceeds this value",
    )
    parser.add_argument(
        "--drift-detection-threshold",
        type=float,
        default=0.05,
        help="Per-feature statistical threshold used by Evidently",
    )
    arguments = parser.parse_args()

    exit_code = monitor_drift(
        dataset_name=arguments.dataset_config,
        report_path=Path(arguments.report_path),
        max_drift_share=arguments.max_drift_share,
        drift_detection_threshold=arguments.drift_detection_threshold,
    )
    sys.exit(exit_code)
