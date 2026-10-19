# scifair-toolkit

A small, open-source Python toolkit for designing and analyzing controlled science fair experiments: randomized trial sequencing, statistical comparison across conditions, and basic power analysis. Extracted and generalized from the experimental pipeline built for a real ISEF-track research project comparing motor control strategies on an autonomous robot.

This isn't specific to robotics. Any science fair project comparing multiple conditions (treatments, algorithms, materials, etc.) across repeated trials can use it.

## Why this exists

A lot of good high school research gets weakened by avoidable methodology gaps: ad hoc trial ordering (no protection against confounds like equipment drift or fatigue), no formal statistical comparison beyond eyeballing averages, and no check on whether the sample size is even large enough to detect a real effect. This toolkit makes the "boring but important" parts of good experimental design easy to do correctly.

## Features

- **Randomized trial sequencing** (`scifair_toolkit.randomize`): generate a fixed, reproducible randomized order for any set of experimental conditions, so trial order doesn't confound your results.
- **Statistical comparison** (`scifair_toolkit.stats`): one-way ANOVA across conditions, Bonferroni-corrected pairwise t-tests, and summary statistics tables, with clear plain-language output.
- **Power analysis helper** (`scifair_toolkit.stats.estimate_required_sample_size`): given a preliminary variance estimate, calculates roughly how many repetitions per condition you need to reliably detect a meaningful effect.
- **Comparison plotting** (`scifair_toolkit.plotting`): quick boxplot comparisons across conditions.

## Installation

```bash
git clone https://github.com/Kero105/scifair-toolkit.git
cd scifair-toolkit
pip install -e .
```

## Quick start

```python
from scifair_toolkit.randomize import generate_trial_sequence
from scifair_toolkit.stats import run_anova, run_pairwise_ttests, summary_table
from scifair_toolkit.plotting import plot_comparison
import pandas as pd

# 1. Generate your randomized trial order BEFORE you start collecting data
conditions = {
    "treatment": ["A", "B", "C"],
    "surface": ["wood", "tile"],
}
sequence = generate_trial_sequence(conditions, reps_per_condition=8, seed=2026)
sequence.to_csv("my_trial_sequence.csv", index=False)
# -> follow this exact order when you actually run trials

# 2. After collecting data, load it and analyze
df = pd.read_csv("my_results.csv")  # needs a column per condition variable + your outcome measure

summary_table(df, group_col="treatment", value_col="outcome")
run_anova(df, group_col="treatment", value_col="outcome")
run_pairwise_ttests(df, group_col="treatment", value_col="outcome")
plot_comparison(df, group_col="treatment", value_col="outcome", save_path="comparison.png")
```

## Power analysis

```python
from scifair_toolkit.stats import estimate_required_sample_size

# Using variance from a small dry-run batch
n_needed = estimate_required_sample_size(
    preliminary_std=0.4,
    detectable_effect=0.5,  # smallest difference between group means you care about detecting
    alpha=0.05,
    power=0.80,
)
print(f"You need at least {n_needed} repetitions per condition.")
```

## Tests

The tests check the statistics against SciPy and known textbook values, and make sure the trial sequences come out balanced and reproducible.

```bash
pip install -e . pytest
pytest
```

## Background

Built as part of an independent research project on predictive electrical feedback control for energy-efficient autonomous mobile robots, comparing five motor control strategies (fixed-speed, PID, threshold-based protection, predictive, and hybrid) using randomized trial ordering and formal statistical comparison across terrain and payload conditions.

## License

MIT. Use it for your own project, science fair or otherwise.

## Contributing

Issues and pull requests welcome, especially from other students building out their own research methodology.
