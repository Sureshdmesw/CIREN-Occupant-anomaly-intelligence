import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PHASE 9D — OCCUPANT STATE ANOMALY SCORING
# ============================================================

root = Path(
    r"D:\Datasets\occupant_position_ground_truth"
)

phase2 = root / "phase2_analysis"

events_file = (
    root /
    "phase9_event_generation" /
    "synthetic_occupant_events_10000.csv"
)

out = (
    root /
    "phase9_anomaly_scoring"
)

events = pd.read_csv(events_file)

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

for c in features:
    events[c] = (
        events[c]
        .fillna("NOT_COLLECTED/UNKNOWN")
        .astype(str)
    )

# ============================================================
# DEPENDENCY STRUCTURE
# ============================================================

structures = {

    "seat_position": (
        [],
        "seat_position"
    ),

    "occupant_role": (
        ["seat_position"],
        "occupant_role"
    ),

    "seat_track_position": (
        ["seat_position", "occupant_role"],
        "seat_track_position"
    ),

    "seat_type": (
        ["seat_position", "seat_track_position"],
        "seat_type"
    ),

    "posture": (
        [
            "seat_position",
            "seat_track_position",
            "seat_type"
        ],
        "posture"
    ),

    "belt_use": (
        [
            "seat_position",
            "occupant_role",
            "seat_track_position",
            "seat_type"
        ],
        "belt_use"
    ),

    "belt_anchorage": (
        [
            "seat_type",
            "belt_use"
        ],
        "belt_anchorage"
    ),

    "belt_failure": (
        ["belt_use"],
        "belt_failure"
    ),

    "belt_availability": (
        [
            "belt_use",
            "belt_anchorage"
        ],
        "belt_availability"
    ),

    "seat_performance": (
        [
            "seat_position",
            "seat_track_position",
            "seat_type"
        ],
        "seat_performance"
    )
}

# ============================================================
# LOAD EMPIRICAL CONDITIONAL TABLES
# ============================================================

tables = {}

for name, (
    parents,
    target
) in structures.items():

    filename = {
        "seat_position":
            "P_seat_position.csv",

        "occupant_role":
            "P_role_given_seat_position.csv",

        "seat_track_position":
            "P_track_given_seat_role.csv",

        "seat_type":
            "P_type_given_seat_track.csv",

        "posture":
            "P_posture_given_position_track_type.csv",

        "belt_use":
            "P_belt_use_given_position_role_track_type.csv",

        "belt_anchorage":
            "P_anchorage_given_type_use.csv",

        "belt_failure":
            "P_failure_given_belt_use.csv",

        "belt_availability":
            "P_availability_given_use_anchorage.csv",

        "seat_performance":
            "P_seat_performance_given_position_track_type.csv"
    }[name]

    table = pd.read_csv(
        phase2 / filename
    )

    tables[name] = table

# ============================================================
# PROBABILITY LOOKUP
# ============================================================

def get_probability(
    table,
    parents,
    target,
    row
):

    if parents:

        mask = np.ones(
            len(table),
            dtype=bool
        )

        for p in parents:

            mask &= (
                table[p].astype(str)
                == str(row[p])
            )

        mask &= (
            table[target].astype(str)
            == str(row[target])
        )

    else:

        mask = (
            table[target].astype(str)
            == str(row[target])
        )

    subset = table[mask]

    if len(subset) == 0:
        return 0.0

    return float(
        subset.iloc[0]["probability"]
    )

# ============================================================
# SCORE EVENTS
# ============================================================

factor_probability_columns = []
factor_score_columns = []

for name, (
    parents,
    target
) in structures.items():

    probability_column = (
        name +
        "_probability"
    )

    score_column = (
        name +
        "_surprisal_bits"
    )

    factor_probability_columns.append(
        probability_column
    )

    factor_score_columns.append(
        score_column
    )

    probabilities = []

    for _, row in events.iterrows():

        p = get_probability(
            tables[name],
            parents,
            target,
            row
        )

        probabilities.append(p)

    events[probability_column] = probabilities

    events[score_column] = np.where(
        events[probability_column] > 0,
        -np.log2(
            events[probability_column]
        ),
        np.inf
    )

# ============================================================
# TOTAL OCCUPANT SURPRISAL
# ============================================================

events["occupant_anomaly_score_bits"] = (
    events[factor_score_columns]
    .sum(axis=1)
)

# ============================================================
# EXPECTED PROBABILITY
# ============================================================

events["joint_probability"] = (
    events[factor_probability_columns]
    .prod(axis=1)
)

# ============================================================
# DOMINANT CONTRIBUTOR
# ============================================================

score_matrix = events[
    factor_score_columns
].copy()

score_matrix.columns = [
    c.replace(
        "_surprisal_bits",
        ""
    )
    for c in score_matrix.columns
]

events["dominant_contributor"] = (
    score_matrix
    .idxmax(axis=1)
)

events["dominant_contributor_score_bits"] = (
    score_matrix.max(axis=1)
)

# ============================================================
# EMPIRICAL PERCENTILE
# ============================================================

events["occupant_anomaly_percentile"] = (
    events[
        "occupant_anomaly_score_bits"
    ]
    .rank(
        method="average",
        pct=True
    )
    * 100
)

# ============================================================
# ANOMALY TIERS
# ============================================================

def classify(percentile):

    if percentile < 50:
        return "BASELINE"

    if percentile < 75:
        return "MODERATE"

    if percentile < 90:
        return "ELEVATED"

    if percentile < 97:
        return "HIGH"

    return "EXTREME"

events["anomaly_tier"] = (
    events[
        "occupant_anomaly_percentile"
    ]
    .apply(classify)
)

# ============================================================
# MODEL METADATA
# ============================================================

events["model_type"] = (
    "DEPENDENCY_AWARE_EMPIRICAL"
)

events["model_version"] = (
    "CIREN-DAM-v1"
)

events["source_type"] = (
    "SYNTHETIC_FROM_CIREN_EMPIRICAL_MODEL"
)

events["source_reference"] = (
    "NHTSA CIREN Public Data v1.2.1 / OA"
)

events["geometry_status"] = (
    "NOT_PROVIDED"
)

events["temporal_status"] = (
    "SYNTHETIC_EVENT_TIME"
)

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([

    [
        "events",
        len(events)
    ],

    [
        "mean_anomaly_score_bits",
        events[
            "occupant_anomaly_score_bits"
        ].mean()
    ],

    [
        "median_anomaly_score_bits",
        events[
            "occupant_anomaly_score_bits"
        ].median()
    ],

    [
        "minimum_anomaly_score_bits",
        events[
            "occupant_anomaly_score_bits"
        ].min()
    ],

    [
        "maximum_anomaly_score_bits",
        events[
            "occupant_anomaly_score_bits"
        ].max()
    ],

    [
        "baseline_events",
        (
            events["anomaly_tier"]
            == "BASELINE"
        ).sum()
    ],

    [
        "moderate_events",
        (
            events["anomaly_tier"]
            == "MODERATE"
        ).sum()
    ],

    [
        "elevated_events",
        (
            events["anomaly_tier"]
            == "ELEVATED"
        ).sum()
    ],

    [
        "high_events",
        (
            events["anomaly_tier"]
            == "HIGH"
        ).sum()
    ],

    [
        "extreme_events",
        (
            events["anomaly_tier"]
            == "EXTREME"
        ).sum()
    ]

],
columns=[
    "metric",
    "value"
])

# ============================================================
# CONTRIBUTOR SUMMARY
# ============================================================

contributors = (
    events[
        factor_score_columns
    ]
    .mean()
    .sort_values(
        ascending=False
    )
    .rename(
        "mean_surprisal_bits"
    )
    .reset_index()
)

contributors.columns = [
    "feature",
    "mean_surprisal_bits"
]

contributors[
    "percentage_of_total_mean_surprisal"
] = (
    contributors[
        "mean_surprisal_bits"
    ]
    /
    contributors[
        "mean_surprisal_bits"
    ].sum()
    * 100
)

# ============================================================
# SAVE
# ============================================================

event_file = (
    out /
    "scored_occupant_events_10000.csv"
)

summary_file = (
    out /
    "phase9D_anomaly_summary.csv"
)

contributors_file = (
    out /
    "phase9D_contributor_summary.csv"
)

events.to_csv(
    event_file,
    index=False
)

summary.to_csv(
    summary_file,
    index=False
)

contributors.to_csv(
    contributors_file,
    index=False
)

# ============================================================
# REPORT
# ============================================================

print("\n==============================================")
print("PHASE 9D — OCCUPANT ANOMALY SCORING")
print("==============================================")

print(
    "\nEVENTS:",
    len(events)
)

print(
    "\nANOMALY SCORE STATISTICS:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nANOMALY TIERS:"
)

print(
    events[
        "anomaly_tier"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nTOP DOMINANT CONTRIBUTORS:"
)

print(
    events[
        "dominant_contributor"
    ]
    .value_counts()
    .head(15)
    .to_string()
)

print(
    "\nMEAN SURPRISAL CONTRIBUTION:"
)

print(
    contributors.to_string(
        index=False
    )
)

print("\nFILES CREATED:")

print(event_file)
print(summary_file)
print(contributors_file)

