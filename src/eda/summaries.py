"""Tabular data-quality and target-distribution summaries."""

import pandas as pd


def summarize_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return column-level data-quality statistics without printing them."""
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.to_numpy(),
        "Missing Values": df.isna().sum().to_numpy(),
        "Missing %": df.isna().mean().mul(100).round(2).to_numpy(),
        "Unique Values": df.nunique().to_numpy(),
    })


def summarize_target(values: pd.Series) -> pd.DataFrame:
    """Return target counts and percentages, matching the notebook's rounding."""
    return pd.DataFrame({
        "count": values.value_counts(),
        "percent": values.value_counts(normalize=True).round(4).mul(100),
    })
