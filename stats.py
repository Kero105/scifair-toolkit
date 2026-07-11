"""
Statistical comparison helpers for controlled experiments: summary stats,
ANOVA, pairwise t-tests, and a simple power-analysis estimator.
"""

import math
import pandas as pd
from scipy import stats as scipy_stats


def summary_table(df: pd.DataFrame, group_col: str, value_col: str, group_order: list = None) -> pd.DataFrame:
    """Prints and returns mean/std/count of value_col, grouped by group_col."""
    summary = df.groupby(group_col)[value_col].agg(["mean", "std", "count"])
    if group_order:
        summary = summary.reindex(group_order)
    print(f"\n=== Summary statistics: {value_col} by {group_col} ===")
    print(summary.round(4))
    return summary


def run_anova(df: pd.DataFrame, group_col: str, value_col: str, group_order: list = None):
    """One-way ANOVA testing whether value_col differs across levels of group_col."""
    groups_list = group_order if group_order else df[group_col].unique()
    groups = [df[df[group_col] == g][value_col].dropna().values for g in groups_list]
    groups = [g for g in groups if len(g) > 0]

    f_stat, p_value = scipy_stats.f_oneway(*groups)
    print(f"\n=== One-way ANOVA: {value_col} across {group_col} ===")
    print(f"F-statistic: {f_stat:.4f}   p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("Result: statistically significant difference (p < 0.05).")
    else:
        print("Result: not statistically significant at alpha=0.05.")
    return f_stat, p_value


def run_pairwise_ttests(df: pd.DataFrame, group_col: str, value_col: str, group_order: list = None):
    """Bonferroni-corrected pairwise t-tests between every pair of group levels."""
    groups_list = group_order if group_order else sorted(df[group_col].unique())
    n_comparisons = len(groups_list) * (len(groups_list) - 1) // 2
    alpha_corrected = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    print(f"\n=== Pairwise t-tests: {value_col} (Bonferroni-corrected, alpha={alpha_corrected:.5f}) ===")
    for i in range(len(groups_list)):
        for j in range(i + 1, len(groups_list)):
            a = df[df[group_col] == groups_list[i]][value_col].dropna()
            b = df[df[group_col] == groups_list[j]][value_col].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            t_stat, p_val = scipy_stats.ttest_ind(a, b)
            sig = "SIGNIFICANT" if p_val < alpha_corrected else "not significant"
            print(f"{str(groups_list[i]):15s} vs {str(groups_list[j]):15s}: "
                  f"t={t_stat:7.3f}  p={p_val:.5f}  ({sig})")


def estimate_required_sample_size(preliminary_std: float, detectable_effect: float,
                                   alpha: float = 0.05, power: float = 0.80,
                                   n_groups: int = 2) -> int:
    """
    Rough sample-size estimate per group needed to detect a given effect size,
    using a standard two-sample formula as a lightweight alternative to a full
    G*Power session. For a more rigorous analysis (especially with >2 groups
    or ANOVA-specific power), cross-check with G*Power directly.

    Parameters
    ----------
    preliminary_std : float
        Estimated standard deviation of your outcome measure, from a small
        dry-run batch of trials.
    detectable_effect : float
        The smallest difference between group means you want to reliably detect.
    alpha : float
        Significance level (default 0.05).
    power : float
        Desired statistical power (default 0.80).
    n_groups : int
        Number of groups being compared (informational only in this simple
        two-sample approximation; for >2 groups this is a conservative estimate).

    Returns
    -------
    int, estimated minimum repetitions needed per condition.
    """
    # Two-sample t-test sample size formula using normal approximation
    # (z-based, adequate for planning purposes at the scale of a science fair project)
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    z_beta = scipy_stats.norm.ppf(power)

    effect_size = detectable_effect / preliminary_std  # Cohen's d
    n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    n_per_group = math.ceil(n_per_group)
    print(f"Estimated Cohen's d: {effect_size:.3f}")
    print(f"Estimated minimum repetitions per condition: {n_per_group}")
    if n_groups > 2:
        print(f"Note: this is a two-group approximation. With {n_groups} groups being "
              f"compared via ANOVA, consider cross-checking with G*Power for a more precise number.")
    return n_per_group
