import sys

import pandas as pd

sys.path.insert(0, "src")

import monitor_drift


def test_get_feature_columns_excludes_dropped_columns():
    config = {
        "numeric_columns": ["Age", "MonthlyIncome"],
        "categorical_columns": ["Department", "OverTime"],
        "features_to_drop": ["Department"],
    }

    result = monitor_drift.get_feature_columns(config)

    assert result == ["Age", "MonthlyIncome", "OverTime"]


def test_introduce_synthetic_drift_changes_expected_columns():
    production_df = pd.DataFrame(
        {
            "MonthlyIncome": [2000.0, 2500.0, 3000.0, 3500.0],
            "DistanceFromHome": [2.0, 4.0, 6.0, 8.0],
            "Age": [25.0, 30.0, 35.0, 40.0],
            "OverTime": ["No", "No", "No", "No"],
            "Department": ["HR", "HR", "Research & Development", "HR"],
        }
    )

    drifted_df = monitor_drift.introduce_synthetic_drift(production_df, random_state=7)

    assert not drifted_df.equals(production_df)
    assert (drifted_df["MonthlyIncome"] != production_df["MonthlyIncome"]).any()
    assert (drifted_df["DistanceFromHome"] != production_df["DistanceFromHome"]).any()
    assert (drifted_df["OverTime"] == "Yes").any()


def test_summarize_drift_returns_count_share_and_columns():
    snapshot_dict = {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "MonthlyIncome",
                    "threshold": 0.05,
                    "method": "psi",
                },
                "value": 0.001,
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "Age",
                    "threshold": 0.05,
                    "method": "psi",
                },
                "value": 0.40,
            },
        ]
    }

    summary = monitor_drift.summarize_drift(snapshot_dict, ["MonthlyIncome", "Age"])

    assert summary["drift_count"] == 1
    assert summary["total_columns"] == 2
    assert summary["drift_share"] == 0.5
    assert summary["drifted_columns"][0]["column"] == "MonthlyIncome"
