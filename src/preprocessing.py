import pandas as pd
import numpy as np


def fill_missing_with_median(df, columns):
    """Fill missing values in specified columns with the column median."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    return df


def normalize_column(df, column, method="min-max"):
    """Normalize a column using min-max or z-score normalization."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    """Column must be numeric."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError("fColumn '{column}' must be numeric in order to be normalized")

    col_data = df[column]

    if method == "min-max":
        min_val = col_data.min()
        max_val = col_data.max()
        if min_val == max_val:
            df[column] = 0.0
        else:
            df[column] = (col_data - min_val) / (max_val - min_val)
    elif method == "z-score":
        mean_val = col_data.mean()
        std_val = col_data.std()
        if std_val == 0:
            df[column] = 0.0
        else:
            df[column] = (col_data - mean_val) / std_val
    else:
        raise ValueError(f"Unknown method: {method}. Use 'min-max' or 'z-score'")

    return df


def encode_binary_column(df, column, positive_value):
    """Convert a categorical column with two values into 0/1."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    unique_vals = df[column].dropna().unique()
    if len(unique_vals) > 2:
        raise ValueError(
            f"Column '{column}' has {len(unique_vals)} unique values, expected 2"
        )

    df[column] = (df[column] == positive_value).astype(int)
    return df


def create_bins(df, column, bins, labels):
    """Bin a continuous column into categories."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    if len(labels) != len(bins) - 1:
        raise ValueError(f"Number of labels must be one less than the number of bins")

    df[f"{column}_bin"] = pd.cut(df[column], bins=bins, labels=labels, right=False)
    return df


def remove_outliers(df, column, method="iqr", threshold=1.5):
    """Remove rows where a column's value is an outlier."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    if method == "iqr":
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        df = df[(df[column] >= lower) & (df[column] <= upper)]
    elif method == "zscore":
        mean = df[column].mean()
        std = df[column].std()
        df = df[abs(df[column] - mean) <= threshold * std]
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


def validate_dataframe(df, required_columns, target_column):
    """Check that a dataframe meets basic requirements."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if len(df) == 0:
        raise ValueError("Dataframe is empty")

    return True


def clean_data(df, numeric_columns, categorical_columns):
    """Clean a dataframe by handling missing values and encoding categoricals."""
    df = df.copy()

    # Fill numeric missing values with median
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    for col in categorical_columns:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    return df


def encode_categoricals(df, columns):
    """One-hot encode categorical columns."""
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
    return df


def check_data_quality(df, numeric_columns):
    """Return a dictionary of data quality metrics."""
    report = {
        "total_rows": len(df),
        "total_nulls": int(df.isnull().sum().sum()),
        "null_percentage": round(
            df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2
        ),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in numeric_columns:
        if col in df.columns:
            report[f"{col}_min"] = float(df[col].min())
            report[f"{col}_max"] = float(df[col].max())

    return report
