from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.CSV"


def load_actual_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def test_dataset_contains_expected_columns():
    df = load_actual_dataset()

    expected_columns = {
        "Age",
        "Attrition",
        "BusinessTravel",
        "Department",
        "DistanceFromHome",
        "EmployeeNumber",
        "MonthlyIncome",
        "OverTime",
        "TotalWorkingYears",
    }

    assert expected_columns.issubset(df.columns)


def test_dataset_target_contains_only_expected_values():
    df = load_actual_dataset()

    target_values = set(df["Attrition"].dropna().unique())

    assert target_values == {"Yes", "No"}


def test_dataset_numeric_features_are_within_expected_ranges():
    df = load_actual_dataset()

    assert df["Age"].dropna().between(18, 60).all()
    assert df["DailyRate"].dropna().between(102, 1499).all()
    assert df["DistanceFromHome"].dropna().between(1, 29).all()
    assert df["MonthlyIncome"].dropna().between(1009, 19999).all()
    assert df["PercentSalaryHike"].dropna().between(11, 25).all()
    assert df["TotalWorkingYears"].dropna().between(0, 40).all()
