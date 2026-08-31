import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

base = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase2_analysis"
)

src = base / "ciren_empirical_occupant_state.csv"

out_file = (
    base / "conditional_surprisal_records.csv"
)

summary_file = (
    base / "conditional_surprisal_summary.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(src)

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

for c in features:
    df[c] = (
        df[c]
        .fillna("NOT_COLLECTED/UNKNOWN")
        .astype(str)
    )


# ============================================================
# DEPENDENCY STRUCTURE
# ============================================================

structures = {

    "seat_position": [],

    "occupant_role": [
        "seat_position"
    ],

    "seat_track_position": [
        "seat_position",
        "occupant_role"
    ],

    "seat_type": [
        "seat_position",
        "seat_track_position"
    ],

    "posture": [
        "seat_position",
        "seat_track_position",
        "seat_type"
    ],

    "belt_use": [
        "seat_position",
        "occupant_role",
        "seat_track_position",
        "seat_type"
    ],

    "belt_anchorage": [
        "seat_type",
        "belt_use"
    ],

    "belt_failure": [
        "belt_use"
    ],

    "belt_availability": [
        "belt_use",
        "belt_anchorage"
    ],

    "seat_performance": [
        "seat_position",
        "seat_track_position",
        "seat_type"
    ]
}


# ============================================================
# CONDITIONAL PROBABILITY FUNCTION
# ============================================================

def build_probability_table(data, parents, target):

    cols = parents + [target]

    counts = (
        data
        .groupby(
            cols,
            dropna=False
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    if parents:

        totals = (
            counts
            .groupby(parents)["count"]
            .transform("sum")
        )

    else:

        totals = counts["count"].sum()

    counts["probability"] = (
        counts["count"] / totals
    )

    return counts


# ============================================================
# CALCULATE CONDITIONAL SURPRISAL
# ============================================================

all_scores = []

for target, parents in structures.items():

    table = build_probability_table(
        df,
        parents,
        target
    )

    # --------------------------------------------------------
    # Create lookup
    # --------------------------------------------------------

    lookup = {}

    for _, row in table.iterrows():

        key = tuple(
            str(row[c])
            for c in parents + [target]
        )

        lookup[key] = float(
            row["probability"]
        )

    # --------------------------------------------------------
    # Score every observation
    # --------------------------------------------------------

    probabilities = []

    for _, row in df.iterrows():

        key = tuple(
            str(row[c])
            for c in parents + [target]
        )

        probabilities.append(
            lookup.get(key, 0.0)
        )

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    # --------------------------------------------------------
    # No zero probabilities expected for full-data scoring
    # --------------------------------------------------------

    probabilities = np.maximum(
        probabilities,
        1e-15
    )

    surprisal = -np.log2(
        probabilities
    )

    # --------------------------------------------------------
    # Store node-level scores
    # --------------------------------------------------------

    node_result = pd.DataFrame({

        "record_index":
            np.arange(len(df)),

        "feature":
            target,

        "conditional_probability":
            probabilities,

        "conditional_surprisal_bits":
            surprisal
    })

    all_scores.append(
        node_result
    )


# ============================================================
# COMBINE NODE SCORES
# ============================================================

scores = pd.concat(
    all_scores,
    ignore_index=True
)


# ============================================================
# PIVOT TO OBSERVATION × FEATURE
# ============================================================

score_matrix = (
    scores
    .pivot(
        index="record_index",
        columns="feature",
        values="conditional_surprisal_bits"
    )
    .reset_index()
)


# ============================================================
# TOTAL STATE SURPRISAL
# ============================================================

score_features = [
    c for c in score_matrix.columns
    if c != "record_index"
]

score_matrix["total_conditional_surprisal_bits"] = (
    score_matrix[score_features]
    .sum(axis=1)
)


# ============================================================
# NORMALIZED SURPRISAL
# ============================================================

score_matrix["mean_conditional_surprisal_bits"] = (
    score_matrix[
        score_features
    ].mean(axis=1)
)


# ============================================================
# RANK
# ============================================================

score_matrix["surprisal_rank"] = (
    score_matrix[
        "total_conditional_surprisal_bits"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# ============================================================
# SAVE OBSERVATION-LEVEL SCORES
# ============================================================

score_matrix.to_csv(
    out_file,
    index=False
)


# ============================================================
# FEATURE-LEVEL SUMMARY
# ============================================================

feature_summary = []

for feature in score_features:

    values = score_matrix[feature]

    feature_summary.append({

        "feature":
            feature,

        "mean_surprisal_bits":
            values.mean(),

        "std_surprisal_bits":
            values.std(),

        "median_surprisal_bits":
            values.median(),

        "minimum_surprisal_bits":
            values.min(),

        "maximum_surprisal_bits":
            values.max(),

        "p90_surprisal_bits":
            values.quantile(0.90),

        "p95_surprisal_bits":
            values.quantile(0.95),

        "p99_surprisal_bits":
            values.quantile(0.99)
    })


feature_summary = pd.DataFrame(
    feature_summary
).sort_values(
    "mean_surprisal_bits",
    ascending=False
)


# ============================================================
# STATE-LEVEL SUMMARY
# ============================================================

total = score_matrix[
    "total_conditional_surprisal_bits"
]

summary = pd.DataFrame([

    [
        "observations",
        len(df)
    ],

    [
        "mean_total_conditional_surprisal_bits",
        total.mean()
    ],

    [
        "std_total_conditional_surprisal_bits",
        total.std()
    ],

    [
        "median_total_conditional_surprisal_bits",
        total.median()
    ],

    [
        "p90_total_conditional_surprisal_bits",
        total.quantile(0.90)
    ],

    [
        "p95_total_conditional_surprisal_bits",
        total.quantile(0.95)
    ],

    [
        "p99_total_conditional_surprisal_bits",
        total.quantile(0.99)
    ],

    [
        "maximum_total_conditional_surprisal_bits",
        total.max()
    ]

],
columns=[
    "metric",
    "value"
])


# ============================================================
# SAVE SUMMARY
# ============================================================

feature_summary.to_csv(
    summary_file,
    index=False
)

summary.to_csv(
    base / "conditional_surprisal_global_summary.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

print(
    "\n=============================================="
)

print(
    "CONDITIONAL OCCUPANT-STATE SURPRISAL"
)

print(
    "=============================================="
)

print(
    "\nGLOBAL SUMMARY:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nFEATURE-LEVEL SURPRISAL:"
)

print(
    feature_summary.to_string(
        index=False
    )
)

print(
    "\nTOP 20 MOST SURPRISING OBSERVATIONS:"
)

print(
    score_matrix[
        [
            "record_index",
            "total_conditional_surprisal_bits",
            "mean_conditional_surprisal_bits",
            "surprisal_rank"
        ]
    ]
    .sort_values(
        "total_conditional_surprisal_bits",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

print(
    "\nFILES CREATED:"
)

print(out_file)

print(summary_file)

print(
    base /
    "conditional_surprisal_global_summary.csv"
)

