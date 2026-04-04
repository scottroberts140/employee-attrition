import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, "src")

import train


def test_model_validation_predictions_have_correct_shape_and_binary_type(monkeypatch):
    df = pd.DataFrame(
        {
            "Age": [22, 24, 26, 44, 46, 48, 28, 50],
            "MonthlyIncome": [2100, 2200, 2300, 7800, 7900, 8100, 2400, 8200],
            "OverTime": ["Yes", "Yes", "Yes", "No", "No", "No", "Yes", "No"],
            "Department": [
                "Sales",
                "Sales",
                "HR",
                "Research & Development",
                "Research & Development",
                "HR",
                "Sales",
                "Research & Development",
            ],
            "Attrition": ["Yes", "Yes", "Yes", "No", "No", "No", "Yes", "No"],
        }
    )

    monkeypatch.setattr(train, "load_data", lambda url: df)

    config = {
        "data_url_raw": "./unused.csv",
        "features_to_drop": [],
        "numeric_columns": ["Age", "MonthlyIncome"],
        "categorical_columns": ["OverTime", "Department"],
        "target": "Attrition",
        "test_size": 0.25,
        "random_state": 7,
        "model_type": "LR",
        "solver": "liblinear",
        "max_iter": 200,
        "C": 1.0,
    }

    model, X_train, y_train, X_test, y_test = train.train_model(config)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert predictions.ndim == 1
    assert set(predictions.tolist()).issubset({0, 1})


def test_model_validation_meets_minimum_accuracy_on_known_test_set():
    X_train = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 8, 9, 10],
            "monthly_income": [2000, 2200, 2400, 7000, 7200, 7400],
        }
    )
    y_train = pd.Series([1, 1, 1, 0, 0, 0])
    X_test = pd.DataFrame(
        {
            "tenure": [2, 9, 4, 8],
            "monthly_income": [2100, 7300, 2600, 7100],
        }
    )
    y_test = pd.Series([1, 0, 1, 0])

    model = LogisticRegression(solver="liblinear", random_state=7)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.75