import pandas as pd
import numpy as np


def handle_missing_values(
    df, columns, strategy="median", outlier_method="iqr", outlier_threshold=1.5
):
    """Handle missing values in selected columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to process.
    columns : list[str]
        Column names to inspect for missing values.
    strategy : str, default="median"
        Strategy used to handle missing values. Supported values are
        "median", "mean", "mean_median", and "drop".
    outlier_method : str, default="iqr"
        Outlier detection method passed to ``has_outliers`` when
        ``strategy="mean_median"``.
    outlier_threshold : float, default=1.5
        Threshold passed to ``has_outliers`` when
        ``strategy="mean_median"``.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with missing values handled.

    Examples
    --------
    >>> df = pd.DataFrame({"age": [20.0, None, 40.0]})
    >>> handle_missing_values(df, ["age"], strategy="median")
    >>> handle_missing_values(df, ["age"], strategy="mean_median", outlier_method="zscore", outlier_threshold=2.0)
    >>> handle_missing_values(df, ["age"], strategy="drop")
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    # Check strategy for a valid value
    valid_strategies = ["median", "mean", "mean_median", "drop"]

    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy: {strategy}. Supported values are {valid_strategies}."
        )

    def get_fill_value_function(column):
        if strategy == "median":
            return pd.Series.median
        if strategy == "mean":
            return pd.Series.mean
        if strategy == "mean_median":
            if has_outliers(
                df,
                column,
                method=outlier_method,
                threshold=outlier_threshold,
            ):
                return pd.Series.median
            return pd.Series.mean

        raise ValueError(f"Unknown strategy: {strategy}")

    if strategy == "drop":
        df = df.dropna(subset=columns)
    else:
        for col in columns:
            fill_value_function = get_fill_value_function(col)
            fill_value = fill_value_function(df[col])
            df[col] = df[col].fillna(fill_value)

    return df


def normalize_column(df, column, method="min-max"):
    """Normalize a numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to process.
    column : str
        Name of the numeric column to normalize.
    method : str, default="min-max"
        Normalization method. Supported values are "min-max" and "z-score".

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with the normalized column.

    Examples
    --------
    >>> df = pd.DataFrame({"age": [20.0, 30.0, 40.0]})
    >>> normalize_column(df, "age", method="min-max")
    >>> normalize_column(df, "age", method="z-score")
    """
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric in order to be normalized")

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
        raise ValueError(
            f"Unknown method: {method}. Supported values are 'min-max' and 'z-score'."
        )

    return df


def encode_binary_column(df, column, positive_value):
    """Encode a two-value categorical column as 0 and 1.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to process.
    column : str
        Name of the column to encode.
    positive_value : object
        Value that should be encoded as 1. All other non-null values are encoded
        as 0.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with the encoded column.

    Examples
    --------
    >>> df = pd.DataFrame({"Attrition": ["Yes", "No", "Yes"]})
    >>> encode_binary_column(df, "Attrition", "Yes")
    """
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
    """Bin a continuous column into labeled intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to process.
    column : str
        Name of the numeric column to bin.
    bins : list[float]
        Bin edges passed to ``pandas.cut``.
    labels : list[str]
        Labels for the created intervals. The number of labels must be one less
        than the number of bins.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with a new ``<column>_bin`` column.

    Examples
    --------
    >>> df = pd.DataFrame({"salary": [30000, 50000, 80000]})
    >>> create_bins(df, "salary", bins=[0, 40000, 70000, 100000], labels=["low", "mid", "high"])
    """
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    if len(labels) != len(bins) - 1:
        raise ValueError("Number of labels must be one less than the number of bins")

    df[f"{column}_bin"] = pd.cut(df[column], bins=bins, labels=labels, right=False)
    return df


def remove_outliers(df, column, method="iqr", threshold=1.5):
    """Remove rows containing outliers in a selected column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to process.
    column : str
        Name of the numeric column to evaluate.
    method : str, default="iqr"
        Outlier detection method. Supported values are "iqr" and "zscore".
    threshold : float, default=1.5
        Detection threshold used by the selected method.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with outlier rows removed.

    Examples
    --------
    >>> df = pd.DataFrame({"salary": [50, 52, 53, 200]})
    >>> remove_outliers(df, "salary", method="iqr", threshold=1.5)
    >>> remove_outliers(df, "salary", method="zscore", threshold=2.0)
    """
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
        raise ValueError(
            f"Unknown method: {method}. Supported values are 'iqr' and 'zscore'."
        )

    return df


def has_outliers(df, column, method="iqr", threshold=1.5):
    """Check whether a column contains outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to inspect.
    column : str
        Name of the numeric column to evaluate.
    method : str, default="iqr"
        Outlier detection method. Supported values are "iqr" and "zscore".
    threshold : float, default=1.5
        Detection threshold used by the selected method.

    Returns
    -------
    bool
        ``True`` if outliers are present; otherwise ``False``.

    Examples
    --------
    >>> df = pd.DataFrame({"salary": [50, 52, 53, 200]})
    >>> has_outliers(df, "salary")
    >>> has_outliers(df, "salary", method="zscore", threshold=2.0)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    if method == "iqr":
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outlier_mask = (df[column] < lower) | (df[column] > upper)
    elif method == "zscore":
        mean = df[column].mean()
        std = df[column].std()
        if std == 0:
            return False
        outlier_mask = abs(df[column] - mean) > threshold * std
    else:
        raise ValueError(
            f"Unknown method: {method}. Supported values are 'iqr' and 'zscore'."
        )

    return bool(outlier_mask.any())


def validate_dataframe(df, required_columns, target_column):
    """Validate that a dataframe contains the required structure.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to validate.
    required_columns : list[str]
        Columns that must be present in the dataframe.
    target_column : str
        Name of the target column that must be present.

    Returns
    -------
    bool
        ``True`` when the dataframe is valid.

    Examples
    --------
    >>> df = pd.DataFrame({"age": [25], "salary": [50000], "attrition": [1]})
    >>> validate_dataframe(df, ["age", "salary"], "attrition")
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    if len(df) == 0:
        raise ValueError("Dataframe is empty")

    return True


def clean_data(df, numeric_columns, categorical_columns):
    """Clean a dataframe by filling missing numeric and categorical values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to clean.
    numeric_columns : list[str]
        Numeric columns whose missing values should be filled with the median.
    categorical_columns : list[str]
        Categorical columns whose missing values should be filled with the mode.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with missing values filled.

    Examples
    --------
    >>> df = pd.DataFrame({"age": [20.0, None], "dept": ["Sales", None]})
    >>> clean_data(df, numeric_columns=["age"], categorical_columns=["dept"])
    """
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
    """One-hot encode categorical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to process.
    columns : list[str]
        Categorical columns to encode.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with dummy-variable columns added.

    Examples
    --------
    >>> df = pd.DataFrame({"dept": ["Sales", "HR", "Sales"]})
    >>> encode_categoricals(df, ["dept"])
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
    return df


def check_data_quality(df, numeric_columns):
    """Summarize basic data quality metrics for a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to inspect.
    numeric_columns : list[str]
        Numeric columns for which minimum and maximum values should be reported.

    Returns
    -------
    dict
        Dictionary containing row count, null count, null percentage,
        duplicate count, and min/max values for requested numeric columns.

    Examples
    --------
    >>> df = pd.DataFrame({"age": [20, 30], "salary": [50000, 60000]})
    >>> check_data_quality(df, ["age", "salary"])
    """
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
