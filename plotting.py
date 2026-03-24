"""
Plotting utilities: line plots, correlation charts, missing value heatmap.
"""
import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from .data_processing import extract_batch_logic, extract_number


def plot_indexed_lineplots(df, batch_col="BatchID", cols_per_row=3):
    """
    Plot all numeric columns as line plots, x-axis = chronological batch sequence.
    BUGFIX: when num_rows=1, np.array(axes) is 1-D and flatten() works correctly.
    Returns matplotlib Figure or None.
    """
    sorted_df = df.copy()
    if batch_col in sorted_df.columns:
        sorted_df["_sort"] = sorted_df[batch_col].apply(extract_number)
        sorted_df = sorted_df.sort_values("_sort").reset_index(drop=True)
        sorted_df = sorted_df.drop(columns=["_sort"])
    sorted_df["Sequence_Index"] = sorted_df.index + 1

    exclude_cols = [batch_col, "Sequence_Index"]
    numeric_cols = [
        c for c in sorted_df.select_dtypes(include=["number"]).columns
        if c not in exclude_cols
    ]
    if not numeric_cols:
        return None

    num_rows = math.ceil(len(numeric_cols) / cols_per_row)
    fig, axes = plt.subplots(num_rows, cols_per_row,
                             figsize=(cols_per_row * 5, num_rows * 4),
                             squeeze=False)
    axes = axes.flatten()
    sns.set_style("whitegrid")

    for i, col in enumerate(numeric_cols):
        sns.lineplot(data=sorted_df, x="Sequence_Index", y=col,
                     marker="o", color="royalblue", linewidth=1.5, ax=axes[i])
        axes[i].set_title("\n".join(textwrap.wrap(col, width=30)), fontsize=9, pad=8)
        axes[i].set_xlabel("Batch Sequence")
        axes[i].set_ylabel("Value")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


def plot_clean_lineplots(df, batch_col="BatchID", cols_per_row=3):
    """
    Plot numeric columns with x-axis = YYNN batch code.
    BUGFIX: squeeze=False prevents axes shape inconsistency with 1 row.
    Returns matplotlib Figure or None.
    """
    temp_df = df.copy()
    if batch_col in temp_df.columns:
        temp_df["sort_key"] = temp_df[batch_col].apply(extract_batch_logic)
        temp_df = temp_df.sort_values("sort_key")

    numeric_cols = temp_df.select_dtypes(include=["number"]).columns.tolist()
    for c in ["sort_key", "batch_num"]:
        if c in numeric_cols:
            numeric_cols.remove(c)
    if not numeric_cols:
        return None

    num_rows = math.ceil(len(numeric_cols) / cols_per_row)
    fig, axes = plt.subplots(num_rows, cols_per_row,
                             figsize=(cols_per_row * 5, num_rows * 4),
                             squeeze=False)
    axes = axes.flatten()
    sns.set_style("whitegrid")

    for i, col in enumerate(numeric_cols):
        sns.lineplot(data=temp_df, x="sort_key", y=col,
                     marker="o", color="teal", ax=axes[i])
        axes[i].xaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
        axes[i].set_title("\n".join(textwrap.wrap(col, width=30)), fontsize=9, pad=8)
        axes[i].set_xlabel("Batch (YYNN)")
        axes[i].set_ylabel("Value")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    return fig


def plot_correlation_bar(corr_rank, target_col, top_n, method):
    """Return a seaborn barplot figure of top correlated features."""
    fig, ax = plt.subplots(figsize=(10, 6))
    top_corr = corr_rank.head(top_n)
    sns.barplot(data=top_corr, x="Correlation", y="Feature", palette="vlag", ax=ax)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_title(f"Top {top_n} Features Correlated with\n{target_col}", fontsize=12)
    ax.set_xlabel(f"{method.capitalize()} Correlation Coefficient")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig


def plot_missing_heatmap(df, col_names):
    """Return a heatmap of missing value patterns."""
    fig, ax = plt.subplots(figsize=(14, 4))
    missing_matrix = df[col_names].isnull().T
    sns.heatmap(missing_matrix, cmap="Reds", cbar=False, ax=ax,
                yticklabels=[c[:40] for c in col_names])
    ax.set_xlabel("Sample Index")
    ax.set_title("Missing Value Pattern")
    plt.tight_layout()
    return fig


def plot_yield_tracking(df, tracked_col, batch_col="BatchID",
                        title_prefix="", color="#2e86ab"):
    """
    Plot a single column (usually Yield-related) over batch sequence.
    Used to show the before/after effect of feature engineering steps.

    Parameters
    ----------
    df           : DataFrame containing batch_col and tracked_col
    tracked_col  : column name to plot on Y-axis
    batch_col    : column used for X-axis ordering
    title_prefix : string prepended to the chart title
    color        : line / marker color
    """
    if tracked_col not in df.columns:
        return None

    plot_df = df[[batch_col, tracked_col]].copy() if batch_col in df.columns \
              else df[[tracked_col]].copy()

    if batch_col in plot_df.columns:
        plot_df["_sort"] = plot_df[batch_col].apply(extract_number)
        plot_df = plot_df.sort_values("_sort").reset_index(drop=True)
        plot_df = plot_df.drop(columns=["_sort"])
    plot_df["_seq"] = range(1, len(plot_df) + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(plot_df["_seq"], plot_df[tracked_col],
            marker="o", color=color, linewidth=1.8, ms=5)

    # Shade missing values
    for i, val in enumerate(plot_df[tracked_col]):
        if pd.isna(val):
            ax.axvspan(i + 0.5, i + 1.5, alpha=0.15, color="#e84855")

    if batch_col in plot_df.columns:
        ax.set_xticks(plot_df["_seq"])
        ax.set_xticklabels(
            [str(b)[-6:] for b in plot_df[batch_col]],
            rotation=90, fontsize=7
        )

    title = f"{title_prefix}  {tracked_col}" if title_prefix else tracked_col
    ax.set_title("\n".join(textwrap.wrap(title, width=80)), fontsize=10)
    ax.set_xlabel("Batch Sequence")
    ax.set_ylabel("Value")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig
