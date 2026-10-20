from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

BOOLEAN_COLORS = {True: "#6EC18F", False: "#EA8686"}


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


def _finish_figure(fig: Figure, ax: Axes, labels: Sequence[str]) -> None:
    """Add a shared legend and reserve space for it and the figure title."""
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2,
    )
    fig.tight_layout(pad=3.0, rect=(0, 0, 1, 0.90))


def plot_boolean_feature_composition(
    df: pd.DataFrame, columns: Sequence[str], target_column: str
) -> Figure:
    """Compare boolean feature composition overall and among target column equal to True."""
    fig, axes = plt.subplots(
        len(columns), 2, figsize=(12, 2 * len(columns)), squeeze=False,
    )
    fig.suptitle("Boolean Features: All Customers vs Churned Customers", fontsize=18)
    churned = df.loc[df[target_column]]

    for row, column in enumerate(columns):
        plot_proportion_bar(axes[row, 0], df[column], f"{column}: all customers")
        plot_proportion_bar(
            axes[row, 1], churned[column], f"{column}: churned customers",
        )

    _finish_figure(fig, axes[0, 0], ["True", "False"])
    return fig


def plot_churn_by_boolean_feature(
    df: pd.DataFrame, columns: Sequence[str], target_column: str
) -> Figure:
    """Compare churn/stay proportions within each False and True group."""
    fig, axes = plt.subplots(
        len(columns), 2, figsize=(12, 2 * len(columns)), squeeze=False,
    )
    fig.suptitle("Churn Rates Within Boolean Feature Groups", fontsize=18)

    for row, column in enumerate(columns):
        for col, value in enumerate((False, True)):
            target = df.loc[df[column].eq(value), target_column]
            plot_proportion_bar(axes[row, col], target, f"{column} = {value}")

    _finish_figure(fig, axes[0, 0], ["Churned", "Stayed"])
    return fig
