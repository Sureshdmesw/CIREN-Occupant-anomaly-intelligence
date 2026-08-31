import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

base = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase2_analysis"
)

state_file = (
    base / "ciren_empirical_occupant_state.csv"
)

surprisal_file = (
    base / "conditional_surprisal_records.csv"
)

out_records = (
    base / "surprisal_decomposition.csv"
)

out_top = (
    base / "top_state_surprisal_decomposition.csv"
)

out_summary = (
    base / "surprisal_contribution_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(state_file)

scores = pd.read_csv(surprisal_file)


# ============================================================
# FEATURES
# ============================================================

features = [
    "seat_position",
    "occupant_role",
    "seat_track_position",
    "posture",
    "seat_type",
    "seat_performance",
    "belt_availability",
    "belt_use",
    "belt_failure",
    "belt_anchorage"
]


# ============================================================
# IDENTIFY SURPRISAL COLUMNS
# ============================================================

surprisal_columns = [
    c for c in scores.columns
    if c in features
]


print(
    "SURPRISAL FEATURES:",
    surprisal_columns
)


# ============================================================
# BUILD DECOMPOSITION
# ============================================================

records = []

for idx, row in scores.iterrows():

    total = row[
        "total_conditional_surprisal_bits"
    ]

    for feature in surprisal_columns:

        contribution = row[feature]

        if total > 0:

            contribution_percent = (
                contribution /
                total *
                100
            )

        else:

            contribution_percent = 0.0

        records.append({

            "record_index":
                int(row["record_index"]),

            "feature":
                feature,

            "surprisal_bits":
                contribution,

            "contribution_percent":
                contribution_percent,

            "total_surprisal_bits":
                total,

            "feature_value":
                df.iloc[
                    int(row["record_index"])
                ][feature]
        })


decomposition = pd.DataFrame(
    records
)


# ============================================================
# SAVE FULL DECOMPOSITION
# ============================================================

decomposition.to_csv(
    out_records,
    index=False
)


# ============================================================
# TOP 100 HIGH-SURPRISAL STATES
# ============================================================

top_indices = (
    scores
    .nlargest(
        100,
        "total_conditional_surprisal_bits"
    )["record_index"]
    .astype(int)
    .tolist()
)

top = decomposition[
    decomposition["record_index"]
    .isin(top_indices)
].copy()


top = top.sort_values(
    [
        "record_index",
        "contribution_percent"
    ],
    ascending=[
        True,
        False
    ]
)


top.to_csv(
    out_top,
    index=False
)


# ============================================================
# FEATURE CONTRIBUTION SUMMARY
# ============================================================

summary = (
    top
    .groupby("feature")
    .agg(
        mean_surprisal_bits=(
            "surprisal_bits",
            "mean"
        ),

        median_surprisal_bits=(
            "surprisal_bits",
            "median"
        ),

        mean_contribution_percent=(
            "contribution_percent",
            "mean"
        ),

        median_contribution_percent=(
            "contribution_percent",
            "median"
        ),

        max_contribution_percent=(
            "contribution_percent",
            "max"
        )
    )
    .reset_index()
)


summary = summary.sort_values(
    "mean_contribution_percent",
    ascending=False
)


summary.to_csv(
    out_summary,
    index=False
)


# ============================================================
# DISPLAY TOP STATES
# ============================================================

print(
    "\n=============================================="
)

print(
    "SURPRISAL DECOMPOSITION"
)

print(
    "=============================================="
)


for record in top_indices[:20]:

    state = top[
        top["record_index"] == record
    ]

    state = state.sort_values(
        "contribution_percent",
        ascending=False
    )

    total = state[
        "total_surprisal_bits"
    ].iloc[0]

    print(
        "\n----------------------------------------------"
    )

    print(
        "RECORD:",
        record
    )

    print(
        "TOTAL SURPRISAL:",
        round(total, 6),
        "bits"
    )

    print(
        state[
            [
                "feature",
                "feature_value",
                "surprisal_bits",
                "contribution_percent"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n=============================================="
)

print(
    "TOP-100 FEATURE CONTRIBUTION SUMMARY"
)

print(
    "=============================================="
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nFILES CREATED:"
)

print(out_records)
print(out_top)
print(out_summary)

