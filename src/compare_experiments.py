import argparse

import mlflow
import pandas as pd

from experiment import (
    configure_mlflow_tracking,
    get_scenario_title,
    print_analysis_report,
    summarize_run,
    summarize_runs_by_model_type,
)


def resolve_experiment_name(
    experiment_name: str | None,
    suite_name: str | None,
    scenario_name: str | None,
) -> str:
    """Resolve the MLflow experiment name from CLI arguments.

    Parameters
    ----------
    experiment_name : str | None
        Explicit MLflow experiment name.
    suite_name : str | None
        Suite filename without the ``.yaml`` extension.
    scenario_name : str | None
        Scenario name inside the suite definition.

    Returns
    -------
    str
        Experiment name to query.
    """
    if experiment_name:
        return experiment_name

    if suite_name and scenario_name:
        return get_scenario_title(suite_name, scenario_name)

    raise ValueError("Provide --experiment-name or both --suite and --scenario.")


def get_ranked_runs(experiment_name: str, metric_name: str) -> pd.DataFrame:
    """Return finished MLflow runs ranked by a metric.

    This wrapper script intentionally uses ``mlflow.search_runs()`` directly to
    satisfy the project requirement for programmatic experiment comparison.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    metric_name : str
        Metric key used for ranking.

    Returns
    -------
    pandas.DataFrame
        Ranked MLflow runs with non-null metric values.
    """
    configure_mlflow_tracking()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        available_names = [exp.name for exp in mlflow.search_experiments()]
        available_display = ", ".join(available_names) if available_names else "none"
        raise ValueError(
            f"MLflow experiment not found: {experiment_name}. "
            f"Available experiments: {available_display}"
        )

    metric_column = f"metrics.{metric_name}"
    runs = pd.DataFrame(
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=[f"{metric_column} DESC"],
        )
    )

    if runs.empty:
        raise ValueError(f"No finished runs found for experiment '{experiment_name}'")

    if metric_column not in runs.columns:
        raise ValueError(
            f"Metric '{metric_name}' was not logged for experiment '{experiment_name}'"
        )

    ranked_runs = runs[runs[metric_column].notna()].copy()
    if ranked_runs.empty:
        raise ValueError(
            f"No finished runs with metric '{metric_name}' were found for "
            f"experiment '{experiment_name}'"
        )

    return ranked_runs.sort_values(metric_column, ascending=False).reset_index(
        drop=True
    )


def compare_experiments(
    experiment_name: str,
    metric_name: str,
    top_n: int = 5,
) -> dict:
    """Build a comparison report for one MLflow experiment.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    metric_name : str
        Metric key used for ranking and comparison.
    top_n : int, default=5
        Number of leading runs to include.

    Returns
    -------
    dict
        Report payload containing top runs, the best run, and model summary.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    runs = get_ranked_runs(experiment_name, metric_name)
    top_runs = [
        summarize_run(row, metric_name) for _, row in runs.head(top_n).iterrows()
    ]

    return {
        "experiment_name": experiment_name,
        "metric_name": metric_name,
        "run_count": len(runs),
        "top_runs": top_runs,
        "best_run": summarize_run(runs.iloc[0], metric_name),
        "model_summary": summarize_runs_by_model_type(runs, metric_name),
    }


def main() -> int:
    """Parse CLI arguments and print the experiment comparison report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        help="Explicit MLflow experiment name to compare",
    )
    parser.add_argument(
        "--suite",
        help="Suite name used to resolve the MLflow experiment title",
    )
    parser.add_argument(
        "--scenario",
        help="Scenario name used to resolve the MLflow experiment title",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help="Primary metric used to rank runs, such as f1 or accuracy",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top runs to include in the report",
    )
    args = parser.parse_args()

    try:
        experiment_name = resolve_experiment_name(
            args.experiment_name,
            args.suite,
            args.scenario,
        )
        report = compare_experiments(experiment_name, args.metric, top_n=args.top_n)
    except ValueError as error:
        parser.error(str(error))

    print_analysis_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
