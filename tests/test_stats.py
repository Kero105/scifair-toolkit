"""
Tests for scifair_toolkit.stats and scifair_toolkit.randomize.

Run:
    pip install -e . pytest
    pytest
"""

import math

import pandas as pd
import pytest
from scipy import stats as scipy_stats

from scifair_toolkit.randomize import generate_trial_sequence
from scifair_toolkit.stats import (
    estimate_required_sample_size,
    run_anova,
    run_pairwise_ttests,
    summary_table,
)


@pytest.fixture
def three_groups():
    """Group C is clearly higher than A and B."""
    return pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "value": [10, 11, 9, 10, 10,
                  10, 12, 11, 9, 10,
                  20, 21, 19, 22, 20],
    })


# ---------- summary_table ----------

def test_summary_table_values(three_groups):
    table = summary_table(three_groups, "group", "value")
    assert table.loc["A", "mean"] == pytest.approx(10.0)
    assert table.loc["C", "mean"] == pytest.approx(20.4)
    assert table.loc["B", "count"] == 5


def test_summary_table_respects_order(three_groups):
    table = summary_table(three_groups, "group", "value", group_order=["C", "A", "B"])
    assert list(table.index) == ["C", "A", "B"]


# ---------- run_anova ----------

def test_anova_matches_scipy(three_groups):
    f_stat, p_value = run_anova(three_groups, "group", "value")
    expected_f, expected_p = scipy_stats.f_oneway(
        [10, 11, 9, 10, 10], [10, 12, 11, 9, 10], [20, 21, 19, 22, 20]
    )
    assert f_stat == pytest.approx(expected_f)
    assert p_value == pytest.approx(expected_p)


def test_anova_detects_real_difference(three_groups):
    _, p_value = run_anova(three_groups, "group", "value")
    assert p_value < 0.05


def test_anova_no_difference_for_identical_groups():
    df = pd.DataFrame({"group": ["A"] * 4 + ["B"] * 4,
                       "value": [1, 2, 3, 4, 1, 2, 3, 4]})
    _, p_value = run_anova(df, "group", "value")
    assert p_value == pytest.approx(1.0)


def test_anova_ignores_missing_values():
    df = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"],
                       "value": [1.0, 2.0, None, 5.0, 6.0, 7.0]})
    f_stat, _ = run_anova(df, "group", "value")
    expected_f, _ = scipy_stats.f_oneway([1.0, 2.0], [5.0, 6.0, 7.0])
    assert f_stat == pytest.approx(expected_f)


# ---------- run_pairwise_ttests ----------

def test_pairwise_ttests_bonferroni_output(three_groups, capsys):
    run_pairwise_ttests(three_groups, "group", "value")
    out = capsys.readouterr().out
    # 3 groups -> 3 comparisons -> alpha = 0.05 / 3
    assert "alpha=0.01667" in out
    assert "A               vs C" in out
    a_vs_c = [line for line in out.splitlines() if line.startswith("A") and "vs C" in line][0]
    assert "SIGNIFICANT" in a_vs_c and "not significant" not in a_vs_c


# ---------- estimate_required_sample_size ----------

def test_sample_size_textbook_value():
    # d = 0.5, alpha = 0.05, power = 0.80 -> about 63 per group (normal approximation)
    n = estimate_required_sample_size(preliminary_std=2.0, detectable_effect=1.0)
    z_a = scipy_stats.norm.ppf(0.975)
    z_b = scipy_stats.norm.ppf(0.80)
    assert n == math.ceil(2 * ((z_a + z_b) / 0.5) ** 2)
    assert n == 63


def test_sample_size_shrinks_for_bigger_effects():
    small_effect = estimate_required_sample_size(1.0, 0.5)
    big_effect = estimate_required_sample_size(1.0, 2.0)
    assert big_effect < small_effect


# ---------- generate_trial_sequence ----------

def test_trial_sequence_size_and_balance():
    df = generate_trial_sequence({"treatment": ["A", "B", "C"], "surface": ["wood", "tile"]},
                                 reps_per_condition=4)
    assert len(df) == 3 * 2 * 4
    counts = df.groupby(["treatment", "surface"]).size()
    assert (counts == 4).all()


def test_trial_sequence_is_reproducible():
    conditions = {"treatment": ["A", "B", "C"]}
    first = generate_trial_sequence(conditions, 5, seed=7)
    second = generate_trial_sequence(conditions, 5, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_trial_sequence_run_order():
    df = generate_trial_sequence({"treatment": ["A", "B"]}, 3)
    assert list(df["run_order"]) == list(range(1, 7))
    assert sorted(df["trial_id"]) == list(range(1, 7))
