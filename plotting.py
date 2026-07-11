"""
Quick comparison plotting for experimental results.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison(df: pd.DataFrame, group_col: str, value_col: str,
                     group_order: list = None, save_path: str = "comparison.png"):
    """Boxplot of value_col across levels of group_col, saved to save_path."""
    groups_list = group_order if group_order else sorted(df[group_col].unique())
    data = [df[df[group_col] == g][value_col].dropna().values for g in groups_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=[str(g) for g in groups_list])
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_title(f"{value_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close()
