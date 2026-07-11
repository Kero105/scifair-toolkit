"""
Randomized trial sequence generation.

Generates a fixed, reproducible, randomized order for any set of experimental
conditions, so trial order itself doesn't become a confound (e.g., equipment
drift, fatigue, or environmental changes correlating with when a condition
happened to be tested).
"""

import itertools
import random
import pandas as pd


def generate_trial_sequence(conditions: dict, reps_per_condition: int, seed: int = 42) -> pd.DataFrame:
    """
    Build a randomized trial sequence covering the full factorial combination
    of the given conditions, repeated `reps_per_condition` times each.

    Parameters
    ----------
    conditions : dict
        Maps a condition name to a list of its possible levels.
        Example: {"treatment": ["A", "B", "C"], "surface": ["wood", "tile"]}
    reps_per_condition : int
        Number of repetitions for each unique combination of condition levels.
    seed : int
        Random seed. Set this once, generate your sequence, and treat the
        output as LOCKED for the rest of your data collection -- don't
        regenerate with a different seed mid-experiment.

    Returns
    -------
    pd.DataFrame with columns: run_order, trial_id, <one column per condition>,
    repetition, completed, notes
    """
    random.seed(seed)

    condition_names = list(conditions.keys())
    condition_values = list(conditions.values())

    combinations = list(itertools.product(*condition_values))

    trials = []
    trial_id = 1
    for combo in combinations:
        for rep in range(reps_per_condition):
            trial = {"trial_id": trial_id}
            for name, value in zip(condition_names, combo):
                trial[name] = value
            trial["repetition"] = rep + 1
            trials.append(trial)
            trial_id += 1

    random.shuffle(trials)
    for i, trial in enumerate(trials):
        trial["run_order"] = i + 1

    df = pd.DataFrame(trials)
    df["completed"] = ""
    df["notes"] = ""

    ordered_cols = ["run_order", "trial_id"] + condition_names + ["repetition", "completed", "notes"]
    df = df[ordered_cols].sort_values("run_order").reset_index(drop=True)

    print(f"Generated {len(df)} trials across {len(combinations)} condition combinations "
          f"({reps_per_condition} reps each), seed={seed}.")
    return df


def add_edge_case_trials(base_df: pd.DataFrame, edge_conditions: list, reps_per_condition: int,
                          seed: int = 42) -> pd.DataFrame:
    """
    Appends additional edge-case / robustness trials (e.g., a low-resource
    condition, an unusual scenario) to an already-generated base sequence,
    re-randomizing run order across the combined set.

    Parameters
    ----------
    base_df : pd.DataFrame
        Output of generate_trial_sequence().
    edge_conditions : list of dict
        Each dict specifies one edge-case condition's values for every column
        that exists in base_df (besides run_order/trial_id/repetition/etc).
    reps_per_condition : int
    seed : int

    Returns
    -------
    pd.DataFrame, combined and re-randomized.
    """
    random.seed(seed + 1)  # different sub-seed so edge trials don't mirror the base shuffle pattern

    condition_cols = [c for c in base_df.columns
                       if c not in ("run_order", "trial_id", "repetition", "completed", "notes")]

    next_id = base_df["trial_id"].max() + 1
    edge_trials = []
    for edge in edge_conditions:
        for rep in range(reps_per_condition):
            trial = {"trial_id": next_id}
            for col in condition_cols:
                trial[col] = edge.get(col, "")
            trial["repetition"] = rep + 1
            trial["completed"] = ""
            trial["notes"] = ""
            edge_trials.append(trial)
            next_id += 1

    combined = pd.concat([base_df.drop(columns=["run_order"]), pd.DataFrame(edge_trials)], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    combined.insert(0, "run_order", range(1, len(combined) + 1))

    print(f"Added {len(edge_trials)} edge-case trials. Total sequence: {len(combined)} trials.")
    return combined
