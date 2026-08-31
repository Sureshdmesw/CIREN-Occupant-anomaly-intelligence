import pandas as pd
import numpy as np
from pathlib import Path

base = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase2_analysis"
)

src = base / "top_state_surprisal_decomposition.csv"

df = pd.read_csv(src)

# ============================================================
# CORRECT TOP-100 FEATURE CONTRIBUTION ANALYSIS
# ============================================================

summary = (
    df.groupby("feature")
    .agg(
        appearances=("record_index", "count"),
        mean_contribution_percent=(
            "contribution_percent",
            "mean"
        ),
        median_contribution_percent=(
            "contribution_percent",
            "median"
        ),
        std_contribution_percent=(
            "contribution_percent",
            "std"
        ),
        max_contribution_percent=(
            "contribution_percent",
            "max"
        ),
        mean_surprisal_bits=(
            "surprisal_bits",
            "mean"
        ),
        median_surprisal_bits=(
            "surprisal_bits",
            "median"
        )
    )
    .reset_index()
)

summary["std_contribution_percent"] = (
    summary["std_contribution_percent"]
    .fillna(0)
)

# Percentage of the top-100 records in which the feature
# appears as a decomposition component.
summary["appearance_rate_percent"] = (
    summary["appearances"] / 100 * 100
)

# IMPORTANT:
# Do NOT sum contribution_percent.
#
# The correct aggregate contribution statistic is the
# arithmetic mean across the top-100 decomposition entries.
summary["top100_mean_contribution_percent"] = (
    summary["mean_contribution_percent"]
)

summary = summary.sort_values(
    "mean_contribution_percent",
    ascending=False
)

out = base / "corrected_top100_contribution_summary.csv"

summary.to_csv(
    out,
    index=False
)

# ============================================================
# NORMALIZED CONTRIBUTION PROFILE
# ============================================================

# Rank based on mean contribution.
summary["contribution_rank"] = (
    summary["mean_contribution_percent"]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)

summary = summary.sort_values(
    "contribution_rank"
)

summary.to_csv(
    base / "corrected_top100_contribution_summary.csv",
    index=False
)

# ============================================================
# VALIDATION CHECK
# ============================================================

# Calculate the average contribution across records.
# For each record, the contributions should approximately
# sum to 100% (subject to numerical precision).

record_totals = (
    df.groupby("record_index")[
        "contribution_percent"
    ]
    .sum()
)

print("\n==============================================")
print("CORRECTED TOP-100 CONTRIBUTION ANALYSIS")
print("==============================================")

print(
    "\nNUMBER OF TOP RECORDS:",
    record_totals.nunique(),
    "unique record totals"
)

print(
    "\nMEAN CONTRIBUTION SUM PER RECORD:",
    record_totals.mean()
)

print(
    "MIN CONTRIBUTION SUM:",
    record_totals.min()
)

print(
    "MAX CONTRIBUTION SUM:",
    record_totals.max()
)

print(
    "\nFEATURE CONTRIBUTION PROFILE:"
)

print(
    summary.to_string(index=False)
)

print(
    "\nIMPORTANT:"
)

print(
    "Contribution percentages are aggregated using "
    "MEAN/MEDIAN, not SUM."
)

print(
    "\nFILE CREATED:"
)

print(
    base / "corrected_top100_contribution_summary.csv"
)

