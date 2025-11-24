from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_category_histograms(
    hs_series: pd.Series,
    mix_hs_series: pd.Series,
    ls_series: pd.Series,
    mix_ls_series: pd.Series,
    title_suffix: str,
    x_label: str,
    figsize: Tuple[int, int] = (14, 4),
) -> None:
    """
    Compare categorical distributions for HS vs. Mix_HS and LS vs. Mix_LS.

    Left subplot : HS vs Mix_HS
    Right subplot: LS vs Mix_LS
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # --- Heart ---
    heart_ax = axes[0]
    heart_categories = sorted(
        set(hs_series.dropna().unique()).union(set(mix_hs_series.dropna().unique()))
    )
    heart_x = np.arange(len(heart_categories))
    bar_width = 0.4

    hs_counts = hs_series.value_counts().reindex(heart_categories, fill_value=0)
    mix_hs_counts = mix_hs_series.value_counts().reindex(heart_categories, fill_value=0)

    heart_ax.bar(heart_x - bar_width / 2, hs_counts.values, bar_width, label="HS", alpha=0.7)
    heart_ax.bar(heart_x + bar_width / 2, mix_hs_counts.values, bar_width, label="Mix_HS", alpha=0.7)

    heart_ax.set_xticks(heart_x)
    heart_ax.set_xticklabels(heart_categories, rotation=45, ha="right")
    heart_ax.set_ylabel("Count")
    heart_ax.set_xlabel(x_label)
    heart_ax.set_title(f"Heart: {title_suffix}")
    heart_ax.legend()

    # --- Lung ---
    lung_ax = axes[1]
    lung_categories = sorted(
        set(ls_series.dropna().unique()).union(set(mix_ls_series.dropna().unique()))
    )
    lung_x = np.arange(len(lung_categories))

    ls_counts = ls_series.value_counts().reindex(lung_categories, fill_value=0)
    mix_ls_counts = mix_ls_series.value_counts().reindex(lung_categories, fill_value=0)

    lung_ax.bar(lung_x - bar_width / 2, ls_counts.values, bar_width, label="LS", alpha=0.7)
    lung_ax.bar(lung_x + bar_width / 2, mix_ls_counts.values, bar_width, label="Mix_LS", alpha=0.7)

    lung_ax.set_xticks(lung_x)
    lung_ax.set_xticklabels(lung_categories, rotation=45, ha="right")
    lung_ax.set_xlabel(x_label)
    lung_ax.set_title(f"Lung: {title_suffix}")
    lung_ax.legend()

    plt.tight_layout()
    plt.show()


def crosstab_heatmap(
    ax: plt.Axes,
    x_series: pd.Series,
    y_series: pd.Series,
    title: str,
    x_label: str,
    y_label: str,
    vmax: int = 5,
) -> None:
    """
    Build crosstab of counts and plot as a blue-white heatmap on the given axis.
    """
    table = pd.crosstab(y_series, x_series)

    im = ax.imshow(
        table.values,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticklabels(table.index)

    # Annotate counts
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, table.values[i, j], ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
