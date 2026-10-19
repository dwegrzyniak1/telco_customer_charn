"""Reproduce eda.ipynb's current checks and boolean-feature analysis.

Run with ``python eda_refactored.py``. Tables print to the terminal and the
two figures open through matplotlib. Importing this module runs no analysis.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


DATA_PATH = Path(__file__).resolve().parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC_COLUMNS = ("tenure", "MonthlyCharges", "TotalCharges")
BOOLEAN_COLUMNS = (
    "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
)
TARGET_COLUMN = "Churn"
BOOLEAN_COLORS = {True: "#6EC18F", False: "#EA8686"}


def convert_to_boolean(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy with the specified Yes/No or 0/1 columns converted."""
    result = df.copy()
    mapping = {"Yes": True, "No": False, 0: False, 1: True}
    for column in columns:
        invalid = ~result[column].isin(mapping)
        if invalid.any():
            unexpected = result.loc[invalid, column].unique().tolist()
            raise ValueError(f"Unexpected values in {column!r}: {unexpected}")
        result[column] = result[column].map(mapping).astype(bool)
    return result


def convert_to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy with blanks treated as missing and other values parsed."""
    result = df.copy()
    for column in columns:
        values = result[column].replace(r"^\s*$", np.nan, regex=True)
        result[column] = pd.to_numeric(values, errors="raise")
    return result


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read and convert types, preserving missing charges for inspection."""
    df = pd.read_csv(path)
    df = convert_to_boolean(df, (*BOOLEAN_COLUMNS, TARGET_COLUMN))
    return convert_to_numeric(df, NUMERIC_COLUMNS)


def summarize_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return column-level data-quality statistics without printing them."""
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.to_numpy(),
        "Missing Values": df.isna().sum().to_numpy(),
        "Missing %": df.isna().mean().mul(100).round(2).to_numpy(),
        "Unique Values": df.nunique().to_numpy(),
    })


def fill_missing_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's assumption that missing total charges mean zero.

    In the inspected dataset, all 11 missing charges occur at zero tenure.
    Interpreting these as new customers with no accumulated charges is an
    assumption, rather than a fact established by the data.
    """
    return df.assign(TotalCharges=df["TotalCharges"].fillna(0))


def summarize_target(values: pd.Series) -> pd.DataFrame:
    """Return target counts and percentages, matching the notebook's rounding."""
    return pd.DataFrame({
        "count": values.value_counts(),
        "percent": values.value_counts(normalize=True).round(4).mul(100),
    })


def plot_proportion_bar(ax: Axes, values: pd.Series, title: str) -> None:
    """Draw True/False percentages on an existing axes, with N in the title."""
    counts = values.value_counts().reindex([True, False], fill_value=0)
    total = counts.sum()
    if total == 0:
        raise ValueError("A proportion bar requires non-missing boolean values.")
    percentages = counts.div(total).mul(100)
    left = 0.0

    for value, color in BOOLEAN_COLORS.items():
        width = percentages[value]
        bars = ax.barh(
            0, width, left=left, height=0.8, color=color, label=str(value),
        )
        if counts[value] > 0:
            ax.bar_label(
                bars, labels=[f"{width:.1f}%"], label_type="center", fontsize=14,
            )
        left += width

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Customers (%)")
    ax.set_title(f"{title} (N={len(values):,})")


def finish_figure(fig: Figure, ax: Axes, labels: Sequence[str]) -> None:
    """Add a shared legend and reserve space for it and the figure title."""
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2,
    )
    fig.tight_layout(pad=3.0, rect=(0, 0, 1, 0.90))


def plot_feature_composition(
    df: pd.DataFrame, columns: Sequence[str] = BOOLEAN_COLUMNS,
) -> Figure:
    """Compare feature composition overall and among churned customers."""
    fig, axes = plt.subplots(
        len(columns), 2, figsize=(12, 2 * len(columns)), squeeze=False,
    )
    fig.suptitle("Boolean Features: All Customers vs Churned Customers", fontsize=18)
    churned = df.loc[df[TARGET_COLUMN]]

    for row, column in enumerate(columns):
        plot_proportion_bar(axes[row, 0], df[column], f"{column}: all customers")
        plot_proportion_bar(
            axes[row, 1], churned[column], f"{column}: churned customers",
        )

    finish_figure(fig, axes[0, 0], ["True", "False"])
    return fig


def plot_churn_by_feature(
    df: pd.DataFrame, columns: Sequence[str] = BOOLEAN_COLUMNS,
) -> Figure:
    """Compare churn/stay proportions within each False and True group."""
    fig, axes = plt.subplots(
        len(columns), 2, figsize=(12, 2 * len(columns)), squeeze=False,
    )
    fig.suptitle("Churn Rates Within Boolean Feature Groups", fontsize=18)

    for row, column in enumerate(columns):
        for col, value in enumerate((False, True)):
            target = df.loc[df[column].eq(value), TARGET_COLUMN]
            plot_proportion_bar(axes[row, col], target, f"{column} = {value}")

    finish_figure(fig, axes[0, 0], ["Churned", "Stayed"])
    return fig


def main() -> None:
    """Run the notebook's current analysis in the same order."""
    df = load_data()
    with pd.option_context("display.max_columns", None):
        print("First five rows:")
        print(df.head().to_string())
        print(f"\nNumber of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Duplicated rows: {df.duplicated().sum()}")
        print("\nData quality before filling missing charges:")
        print(summarize_quality(df).to_string(index=False))
        print("\nCustomers with missing total charges:")
        print(df.loc[df["TotalCharges"].isna()].to_string())
        print(f"\nNumber of rows with tenure = 0: {df['tenure'].eq(0).sum()}")

    df = fill_missing_charges(df)
    print("\nTarget distribution:")
    print(summarize_target(df[TARGET_COLUMN]).to_string())

    plot_feature_composition(df)
    plot_churn_by_feature(df)
    plt.show()


if __name__ == "__main__":
    main()
