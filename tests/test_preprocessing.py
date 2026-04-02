import pandas as pd
import numpy as np
import pytest
import sys

sys.path.insert(0, "src")

from preprocessing import (
    handle_missing_values,
    normalize_column,
    encode_binary_column,
    create_bins,
    remove_outliers,
    has_outliers,
    validate_dataframe,
    clean_data,
    encode_categoricals,
    check_data_quality,
)


def test_handle_missing_values_median_replaces_nulls():
    """Median fill should replace NaN values with the column median."""
    df = pd.DataFrame({"age": [20.0, 30.0, np.nan, 40.0, 50.0]})

    result = handle_missing_values(df, ["age"], strategy="median")

    assert (
        result["age"].isna().sum() == 0
    ), "There should be no missing values after filling"
    assert (
        result["age"].iloc[2] == 35.0
    ), "Missing value should be filled with median (35.0)"


def test_handle_missing_values_does_not_modify_original():
    """The original dataframe should not be changed."""
    df = pd.DataFrame({"age": [20.0, np.nan, 40.0]})
    original_null_count = df["age"].isna().sum()

    handle_missing_values(df, ["age"], strategy="median")

    assert (
        df["age"].isna().sum() == original_null_count
    ), "Original dataframe should not be modified"


def test_handle_missing_values_median_handles_no_nulls():
    """If there are no missing values, the data should be unchanged."""
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0]})

    result = handle_missing_values(df, ["age"], strategy="median")

    pd.testing.assert_frame_equal(result, df)


def test_handle_missing_values_median_multiple_columns():
    """Should handle filling multiple columns at once."""
    df = pd.DataFrame(
        {"age": [20.0, np.nan, 40.0], "income": [50000.0, 60000.0, np.nan]}
    )

    result = handle_missing_values(df, ["age", "income"], strategy="median")

    assert result["age"].isna().sum() == 0
    assert result["income"].isna().sum() == 0


def test_handle_missing_values_mean_replaces_nulls_with_mean():
    """Mean fill should replace NaN values with the column mean."""
    df = pd.DataFrame({"age": [20.0, 30.0, np.nan, 40.0]})

    result = handle_missing_values(df, ["age"], strategy="mean")

    assert result["age"].isna().sum() == 0
    assert result["age"].iloc[2] == 30.0


def test_handle_missing_values_mean_median_uses_median_when_outliers_exist():
    """Mean-median strategy should prefer median for columns with outliers."""
    df = pd.DataFrame({"age": [20.0, 21.0, 22.0, np.nan, 100.0]})

    result = handle_missing_values(df, ["age"], strategy="mean_median")

    assert result["age"].isna().sum() == 0
    assert result["age"].iloc[3] == 21.5


def test_handle_missing_values_mean_median_uses_mean_without_outliers():
    """Mean-median strategy should use mean when the column has no outliers."""
    df = pd.DataFrame({"age": [20.0, 30.0, np.nan, 40.0]})

    result = handle_missing_values(df, ["age"], strategy="mean_median")

    assert result["age"].isna().sum() == 0
    assert result["age"].iloc[2] == 30.0


def test_handle_missing_values_mean_median_uses_custom_outlier_settings():
    """Mean-median strategy should pass custom outlier settings through to detection."""
    df = pd.DataFrame({"age": [10.0, 10.0, 10.0, np.nan, 100.0]})

    result = handle_missing_values(
        df,
        ["age"],
        strategy="mean_median",
        outlier_method="zscore",
        outlier_threshold=10.0,
    )

    assert result["age"].isna().sum() == 0
    assert result["age"].iloc[3] == 32.5


def test_handle_missing_values_raises_on_bad_column():
    """Should raise ValueError (invalid column)."""
    df = pd.DataFrame({"age": [20.0, 30.0]})

    with pytest.raises(ValueError, match="not found"):
        handle_missing_values(df, ["nonexistent_column"], strategy="median")


def test_handle_missing_values_drop_removes_rows_with_missing_values():
    """Drop strategy should remove rows with missing values in the specified columns."""
    df = pd.DataFrame(
        {
            "age": [20.0, np.nan, 40.0],
            "income": [50000.0, 60000.0, np.nan],
            "department": ["Sales", "HR", "IT"],
        }
    )

    result = handle_missing_values(df, ["age", "income"], strategy="drop")

    assert len(result) == 1, "Only rows without missing values should remain"
    assert result.iloc[0]["department"] == "Sales"


def test_handle_missing_values_raises_on_invalid_strategy():
    """Should raise ValueError when the strategy is unsupported."""
    df = pd.DataFrame({"age": [20.0, np.nan, 40.0]})

    with pytest.raises(ValueError, match="Unknown strategy"):
        handle_missing_values(df, ["age"], strategy="mode")


def test_normalize_column_raises_on_bad_column():
    """Should raise ValueError (invalid column)."""
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0]})

    with pytest.raises(ValueError, match="not found"):
        normalize_column(df, "nonexistent_column")


def test_normalize_column_raises_on_non_numeric_dtype():
    """Should raise ValueError (column dtype not numeric)."""
    df = pd.DataFrame({"state": ["KY", "OH", "TN"]})

    with pytest.raises(ValueError, match="must be numeric"):
        normalize_column(df, "state")


def test_normalize_column_raises_on_invalid_method():
    """Should raise ValueError (unknown method)."""
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0]})

    with pytest.raises(ValueError, match="Unknown method"):
        normalize_column(df, "age", "xx")


def test_normalize_column_min_max_equal():
    """Should set all values to 0.0."""
    df = pd.DataFrame({"age": [20.0, 20.0, 20.0]})

    result = normalize_column(df, "age", "min-max")

    assert result["age"].sum() == 0.0  # All values should be set to 0.0


def test_normalize_column_min_max_not_equal():
    """Should replace values based on min-max normalization."""
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0]})

    result = normalize_column(df, "age", "min-max")

    assert result["age"].iloc[0] == 0.0, "Age 20.0 should be replaced with 0.0"
    assert result["age"].iloc[1] == 0.5, "Age 30.0 should be replaced with 0.5"
    assert result["age"].iloc[2] == 1.0, "Age 40.0 should be replaced with 1.0"


def test_normalize_column_z_score_std_is_zero():
    """Should replace values based on z-score normalization, using std == 0."""
    df = pd.DataFrame({"age": [20.0, 20.0, 20.0]})

    result = normalize_column(df, "age", "z-score")

    assert result["age"].sum() == 0.0  # All values should be set to 0.0


def test_normalize_column_z_score_std_is_nonzero():
    """Should replace values based on z-score normalization, using std != 0."""
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0]})
    # mean_val = df.age.mean()
    # std_val = df.age.std()
    # print(f"mean_val: {mean_val}")
    # print(f"std_val: {std_val}")
    # print(f"normalized values: {(df.age - mean_val) / std_val}")

    result = normalize_column(df, "age", "z-score")

    assert result["age"].iloc[0] == -1.0, "Age 20.0 should be replaced with -1.0"
    assert result["age"].iloc[1] == 0.0, "Age 30.0 should be replaced with 0.5"
    assert result["age"].iloc[2] == 1.0, "Age 40.0 should be replaced with 1.0"


def test_encode_binary_column_raises_on_bad_column():
    """Should raise ValueError (invalid column)."""
    df = pd.DataFrame({"state": ["KY", "OH"]})

    with pytest.raises(ValueError, match="not found"):
        encode_binary_column(df, "nonexistent_column", "KY")


def test_encode_binary_column_raises_on_unique_value_count_gt_2():
    """Should raise ValueError (more than 2 unique values)."""
    df = pd.DataFrame({"state": ["KY", "OH", "TN"]})

    with pytest.raises(ValueError, match="expected 2"):
        encode_binary_column(df, "state", "KY")


def test_encode_binary_column():
    """Should set KY = 1; OH = 0"""
    df = pd.DataFrame({"state": ["KY", "OH", "OH", "KY"]})

    result = encode_binary_column(df, "state", "KY")

    assert result["state"].iloc[0] == 1, "KY should be replaced with 1"
    assert result["state"].iloc[1] == 0, "OH should be replaced with 0"
    assert result["state"].iloc[2] == 0, "OH should be replaced with 0"
    assert result["state"].iloc[3] == 1, "KY should be replaced with 1"


def test_create_bins_raises_on_bad_column():
    """Should raise ValueError when the requested column is absent."""
    df = pd.DataFrame({"salary": [30000, 45000, 60000]})

    with pytest.raises(ValueError, match="not found in dataframe"):
        create_bins(
            df,
            "age",
            bins=[0, 30, 60],
            labels=["under_30", "30_plus"],
        )


def test_create_bins_raises_on_label_bin_count_mismatch():
    """Should raise ValueError when labels are not one fewer than bins."""
    df = pd.DataFrame({"salary": [30000, 45000, 60000]})

    with pytest.raises(ValueError, match="one less than the number of bins"):
        create_bins(
            df,
            "salary",
            bins=[0, 40000, 80000],
            labels=["low", "mid", "high"],
        )


def test_create_bins_creates_expected_categories():
    """Should create the expected labeled bins for a generic numeric column."""
    df = pd.DataFrame({"salary": [30000, 45000, 60000, 75000]})

    result = create_bins(
        df,
        "salary",
        bins=[0, 40000, 70000, 90000],
        labels=["low", "mid", "high"],
    )

    assert result["salary_bin"].astype(str).tolist() == [
        "low",
        "mid",
        "mid",
        "high",
    ], "Binned categories should match the supplied labels"


def test_create_bins_uses_left_inclusive_intervals():
    """Bin edges should follow the right=False behavior configured in pd.cut."""
    df = pd.DataFrame({"score": [0, 50, 100]})

    result = create_bins(
        df,
        "score",
        bins=[0, 50, 100, 150],
        labels=["low", "mid", "high"],
    )

    assert result["score_bin"].astype(str).tolist() == ["low", "mid", "high"]


def test_create_bins_does_not_modify_original():
    """The original dataframe should not be changed."""
    df = pd.DataFrame({"salary": [30000, 45000, 60000]})

    create_bins(
        df,
        "salary",
        bins=[0, 40000, 70000],
        labels=["low", "high"],
    )

    assert "salary_bin" not in df.columns, "Original dataframe should be unchanged"


def test_remove_outliers_iqr_removes_extreme_values():
    """IQR method should remove rows outside the IQR fence."""
    df = pd.DataFrame({"salary": [50, 52, 53, 51, 49, 200]})  # 200 is an outlier

    result = remove_outliers(df, "salary", method="iqr", threshold=1.5)

    assert 200 not in result["salary"].values, "Outlier should be removed"
    assert len(result) < len(df), "Result should have fewer rows than original"


def test_remove_outliers_does_not_modify_original():
    """The original dataframe should not be changed."""
    df = pd.DataFrame({"salary": [50, 52, 53, 51, 49, 200]})
    original_len = len(df)

    remove_outliers(df, "salary")

    assert len(df) == original_len, "Original dataframe should not be modified"


def test_remove_outliers_zscore_removes_extreme_values():
    """Z-score method should remove rows beyond the threshold standard deviations."""
    df = pd.DataFrame({"salary": [50, 52, 53, 51, 49, 500]})  # 500 is an outlier

    result = remove_outliers(df, "salary", method="zscore", threshold=2.0)

    assert 500 not in result["salary"].values, "Outlier should be removed"


def test_remove_outliers_keeps_inliers():
    """Values within bounds should not be removed."""
    df = pd.DataFrame({"salary": [50, 52, 53, 51, 49]})

    result = remove_outliers(df, "salary", method="iqr")

    assert len(result) == len(df), "No rows should be removed when no outliers exist"


def test_remove_outliers_invalid_column_raises():
    """Should raise ValueError for a column not in the dataframe."""
    df = pd.DataFrame({"salary": [50, 52, 53]})

    with pytest.raises(ValueError, match="not found in dataframe"):
        remove_outliers(df, "nonexistent_column")


def test_remove_outliers_unknown_method_raises():
    """Should raise ValueError for an unrecognized method."""
    df = pd.DataFrame({"salary": [50, 52, 53]})

    with pytest.raises(ValueError, match="Unknown method"):
        remove_outliers(df, "salary", method="percentile")


def test_has_outliers_iqr_returns_true_for_extreme_values():
    """IQR detection should report True when a column contains outliers."""
    df = pd.DataFrame({"salary": [50, 52, 53, 51, 49, 200]})

    result = has_outliers(df, "salary", method="iqr", threshold=1.5)

    assert result is True, "Outlier detection should return True"


def test_has_outliers_iqr_returns_false_when_no_outliers_exist():
    """IQR detection should report False when all values are within bounds."""
    df = pd.DataFrame({"salary": [50, 52, 53, 51, 49]})

    result = has_outliers(df, "salary", method="iqr")

    assert result is False, "Outlier detection should return False"


def test_has_outliers_zscore_returns_true_for_extreme_values():
    """Z-score detection should report True when a column contains outliers."""
    df = pd.DataFrame({"salary": [10, 10, 10, 10, 100]})

    result = has_outliers(df, "salary", method="zscore", threshold=1.5)

    assert result is True, "Outlier detection should return True"


def test_has_outliers_zscore_returns_false_for_constant_column():
    """Constant columns should not report outliers for z-score detection."""
    df = pd.DataFrame({"salary": [50, 50, 50, 50]})

    result = has_outliers(df, "salary", method="zscore")

    assert result is False, "Constant columns should not report outliers"


def test_has_outliers_invalid_column_raises():
    """Should raise ValueError for a column not in the dataframe."""
    df = pd.DataFrame({"salary": [50, 52, 53]})

    with pytest.raises(ValueError, match="not found in dataframe"):
        has_outliers(df, "nonexistent_column")


def test_has_outliers_unknown_method_raises():
    """Should raise ValueError for an unrecognized detection method."""
    df = pd.DataFrame({"salary": [50, 52, 53]})

    with pytest.raises(ValueError, match="Unknown method"):
        has_outliers(df, "salary", method="percentile")


def test_validate_dataframe_returns_true_for_valid_input():
    """Should return True when the dataframe meets all requirements."""
    df = pd.DataFrame({"age": [25, 30], "salary": [50000, 60000], "attrition": [0, 1]})

    result = validate_dataframe(
        df, required_columns=["age", "salary"], target_column="attrition"
    )

    assert result is True, "Should return True for a valid dataframe"


def test_validate_dataframe_raises_on_missing_required_column():
    """Should raise ValueError when a required column is absent."""
    df = pd.DataFrame({"age": [25, 30], "attrition": [0, 1]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(
            df, required_columns=["age", "salary"], target_column="attrition"
        )


def test_validate_dataframe_raises_on_missing_target_column():
    """Should raise ValueError when the target column is absent."""
    df = pd.DataFrame({"age": [25, 30], "salary": [50000, 60000]})

    with pytest.raises(ValueError, match="Target column"):
        validate_dataframe(
            df, required_columns=["age", "salary"], target_column="attrition"
        )


def test_validate_dataframe_raises_on_empty_dataframe():
    """Should raise ValueError when the dataframe has no rows."""
    df = pd.DataFrame({"age": [], "salary": [], "attrition": []})

    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(
            df, required_columns=["age", "salary"], target_column="attrition"
        )


def test_clean_data_fills_numeric_missing_with_median():
    """Numeric NaNs should be filled with the column median."""
    df = pd.DataFrame(
        {"age": [20.0, 30.0, np.nan, 40.0, 50.0], "dept": ["A", "B", "A", "B", "A"]}
    )

    result = clean_data(df, numeric_columns=["age"], categorical_columns=["dept"])

    assert result["age"].isna().sum() == 0, "There should be no missing values in age"
    assert result["age"].iloc[2] == 35.0, "NaN should be filled with median (35.0)"


def test_clean_data_fills_categorical_missing_with_mode():
    """Categorical NaNs should be filled with the column mode."""
    df = pd.DataFrame(
        {"age": [25, 30, 35], "dept": ["Engineering", "Engineering", None]}
    )

    result = clean_data(df, numeric_columns=["age"], categorical_columns=["dept"])

    assert result["dept"].isna().sum() == 0, "There should be no missing values in dept"
    assert (
        result["dept"].iloc[2] == "Engineering"
    ), "NaN should be filled with mode ('Engineering')"


def test_clean_data_does_not_modify_original():
    """The original dataframe should not be changed."""
    df = pd.DataFrame({"age": [20.0, np.nan, 40.0], "dept": ["A", None, "A"]})

    clean_data(df, numeric_columns=["age"], categorical_columns=["dept"])

    assert df["age"].isna().sum() == 1, "Original numeric column should be unchanged"
    assert (
        df["dept"].isna().sum() == 1
    ), "Original categorical column should be unchanged"


def test_clean_data_ignores_columns_not_in_dataframe():
    """Columns listed but absent from the dataframe should be silently skipped."""
    df = pd.DataFrame({"age": [20.0, np.nan, 40.0]})

    result = clean_data(
        df, numeric_columns=["age", "salary"], categorical_columns=["dept"]
    )

    assert (
        result["age"].isna().sum() == 0
    ), "Present numeric column should still be filled"
    assert "salary" not in result.columns, "Absent column should not be added"
    assert "dept" not in result.columns, "Absent categorical column should not be added"


def test_clean_data_no_missing_values_unchanged():
    """A dataframe with no missing values should pass through unchanged."""
    df = pd.DataFrame({"age": [20.0, 30.0, 40.0], "dept": ["A", "B", "A"]})

    result = clean_data(df, numeric_columns=["age"], categorical_columns=["dept"])

    pd.testing.assert_frame_equal(result, df)


def test_encode_categoricals_drops_original_column():
    """The original categorical column should be replaced by dummy columns."""
    df = pd.DataFrame({"dept": ["Sales", "HR", "Sales"], "age": [25, 30, 35]})

    result = encode_categoricals(df, columns=["dept"])

    assert (
        "dept" not in result.columns
    ), "Original column should be removed after encoding"


def test_encode_categoricals_drop_first_removes_one_category():
    """drop_first=True means one fewer dummy column than unique category values."""
    df = pd.DataFrame({"dept": ["Sales", "HR", "IT"], "age": [25, 30, 35]})

    result = encode_categoricals(df, columns=["dept"])

    dummy_cols = [col for col in result.columns if col.startswith("dept_")]
    assert (
        len(dummy_cols) == 2
    ), "Three categories with drop_first should produce 2 dummy columns"


def test_encode_categoricals_dummy_values_are_integers():
    """Encoded columns should contain integer values (0 or 1)."""
    df = pd.DataFrame({"dept": ["Sales", "HR", "Sales"]})

    result = encode_categoricals(df, columns=["dept"])

    dummy_cols = [col for col in result.columns if col.startswith("dept_")]
    for col in dummy_cols:
        assert (
            result[col].isin([0, 1]).all()
        ), f"Column {col} should contain only 0 and 1"


def test_encode_categoricals_does_not_modify_original():
    """The original dataframe should not be changed."""
    df = pd.DataFrame({"dept": ["Sales", "HR", "Sales"], "age": [25, 30, 35]})

    encode_categoricals(df, columns=["dept"])

    assert "dept" in df.columns, "Original dataframe should still have the dept column"


def test_encode_categoricals_preserves_non_encoded_columns():
    """Columns not in the encode list should be unchanged."""
    df = pd.DataFrame({"dept": ["Sales", "HR", "Sales"], "age": [25, 30, 35]})

    result = encode_categoricals(df, columns=["dept"])

    assert "age" in result.columns, "Non-encoded column should be preserved"
    assert result["age"].tolist() == [
        25,
        30,
        35,
    ], "Non-encoded column values should be unchanged"


def test_check_data_quality_returns_expected_keys():
    """Report should contain all expected top-level keys."""
    df = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 70000]})

    result = check_data_quality(df, numeric_columns=["age", "salary"])

    for key in ["total_rows", "total_nulls", "null_percentage", "duplicate_rows"]:
        assert key in result, f"Expected key '{key}' missing from report"


def test_check_data_quality_total_rows():
    """total_rows should equal the number of rows in the dataframe."""
    df = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 70000]})

    result = check_data_quality(df, numeric_columns=[])

    assert result["total_rows"] == 3, "total_rows should equal the number of rows"


def test_check_data_quality_counts_nulls():
    """total_nulls and null_percentage should reflect missing values."""
    df = pd.DataFrame({"age": [25, np.nan, 35], "salary": [50000, 60000, np.nan]})

    result = check_data_quality(df, numeric_columns=[])

    assert result["total_nulls"] == 2, "total_nulls should count all missing values"
    assert result["null_percentage"] == round(
        2 / (3 * 2) * 100, 2
    ), "null_percentage should be correct"


def test_check_data_quality_counts_duplicate_rows():
    """duplicate_rows should count fully duplicated rows."""
    df = pd.DataFrame({"age": [25, 25, 35], "salary": [50000, 50000, 70000]})

    result = check_data_quality(df, numeric_columns=[])

    assert result["duplicate_rows"] == 1, "duplicate_rows should count one duplicate"


def test_check_data_quality_no_duplicates():
    """duplicate_rows should be 0 when all rows are unique."""
    df = pd.DataFrame({"age": [25, 30, 35], "salary": [50000, 60000, 70000]})

    result = check_data_quality(df, numeric_columns=[])

    assert result["duplicate_rows"] == 0, "duplicate_rows should be 0 for unique rows"


def test_check_data_quality_numeric_min_max():
    """Min and max entries should be added for each numeric column."""
    df = pd.DataFrame({"age": [20, 30, 40], "salary": [50000, 60000, 70000]})

    result = check_data_quality(df, numeric_columns=["age", "salary"])

    assert result["age_min"] == 20.0
    assert result["age_max"] == 40.0
    assert result["salary_min"] == 50000.0
    assert result["salary_max"] == 70000.0


def test_check_data_quality_skips_absent_numeric_columns():
    """Numeric columns not in the dataframe should not appear in the report."""
    df = pd.DataFrame({"age": [20, 30, 40]})

    result = check_data_quality(df, numeric_columns=["age", "salary"])

    assert "salary_min" not in result, "Absent column should not add a min entry"
    assert "salary_max" not in result, "Absent column should not add a max entry"
