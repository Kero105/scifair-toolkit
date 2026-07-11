"""
Example: using scifair-toolkit for a generic 2-condition, 3-treatment experiment.
Run this to see the whole pipeline work end to end with synthetic data.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scifair_toolkit.randomize import generate_trial_sequence, add_edge_case_trials
from scifair_toolkit.stats import summary_table, run_anova, run_pairwise_ttests, estimate_required_sample_size
from scifair_toolkit.plotting import plot_comparison


def main():
    # --- Step 1: generate your randomized trial sequence BEFORE collecting data ---
    conditions = {
        "treatment": ["control", "treatment_A", "treatment_B"],
        "surface": ["wood", "tile"],
    }
    sequence = generate_trial_sequence(conditions, reps_per_condition=5, seed=2026)

    edge_conditions = [{"treatment": "treatment_A", "surface": "wood"}]  # example edge case
    full_sequence = add_edge_case_trials(sequence, edge_conditions, reps_per_condition=3, seed=2026)

    print("\nFirst 5 rows of your locked trial sequence:")
    print(full_sequence.head())

    # --- Step 2: (in real use, you'd collect real data following that sequence) ---
    # Here we simulate fake results just to demonstrate the analysis pipeline.
    rng = np.random.default_rng(1)
    true_means = {"control": 10.0, "treatment_A": 8.5, "treatment_B": 7.0}
    fake_results = full_sequence.copy()
    fake_results["outcome"] = fake_results["treatment"].map(true_means) + rng.normal(0, 1.2, len(fake_results))

    # --- Step 3: analyze ---
    summary_table(fake_results, group_col="treatment", value_col="outcome")
    run_anova(fake_results, group_col="treatment", value_col="outcome")
    run_pairwise_ttests(fake_results, group_col="treatment", value_col="outcome")
    plot_comparison(fake_results, group_col="treatment", value_col="outcome", save_path="example_comparison.png")

    # --- Step 4: power analysis, e.g. after a small dry run ---
    print("\n=== Example power analysis ===")
    estimate_required_sample_size(preliminary_std=1.2, detectable_effect=1.5, n_groups=3)


if __name__ == "__main__":
    main()
