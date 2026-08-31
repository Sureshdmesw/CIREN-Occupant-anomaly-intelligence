import pandas as pd
import numpy as np
from pathlib import Path

base = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase2_analysis"
)

out = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase6_anomaly"
)

src = base / "ciren_empirical_occupant_state.csv"

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
# BUILD CONDITIONAL PROBABILITY TABLES
# ============================================================

def build_probability_table(parents, target):

    cols = parents + [target]

    counts = (
        df.groupby(
            cols,
            dropna=False
        )
        .size()
        .reset_index(name="count")
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


tables = {}

for name, (parents, target) in structures.items():

    tables[name] = build_probability_table(
        parents,
        target
    )

# ============================================================
# BACKOFF PROBABILITY
#
# If an exact conditioning context is unseen,
# progressively remove parents.
# ============================================================

def probability_with_backoff(row, parents, target):

    target_value = str(row[target])

    # Try full conditioning context first.
    for level in range(len(parents), -1, -1):

        active_parents = parents[:level]

        search = tables[target]

        if active_parents:

            mask = np.ones(len(search), dtype=bool)

            for p in active_parents:

                mask &= (
                    search[p].astype(str)
                    == str(row[p])
                )

            subset = search[mask]

        else:

            subset = search

        subset = subset[
            subset[target].astype(str)
            == target_value
        ]

        if len(subset) > 0:

            probability = (
                subset["probability"].iloc[0]
            )

            return float(probability), (
                len(parents) - level
            )

    # Extremely defensive fallback.
    return 1e-12, len(parents) + 1


# ============================================================
# FEATURE-LEVEL CONDITIONAL SURPRISAL
# ============================================================

component_rows = []

for idx, row in df.iterrows():

    total_surprisal = 0.0
    total_backoff = 0

    components = {}

    for name, (parents, target) in structures.items():

        probability, backoff = probability_with_backoff(
            row,
            parents,
            target
        )

        probability = max(
            probability,
            1e-12
        )

        surprisal = -np.log2(
            probability
        )

        components[name] = surprisal

        total_surprisal += surprisal
        total_backoff += backoff

    component_rows.append({
        "record_index": idx,
        "total_surprisal_bits": total_surprisal,
        "total_backoff_level": total_backoff,
        **{
            f"{k}_surprisal_bits": v
            for k, v in components.items()
        }
    })


scores = pd.DataFrame(component_rows)

# ============================================================
# MERGE SCORES
# ============================================================

result = pd.concat(
    [
        df.reset_index(drop=True),
        scores.drop(
            columns=["record_index"]
        )
    ],
    axis=1
)

# ============================================================
# ANOMALY SCORE
# ============================================================

result["anomaly_score_bits"] = (
    result["total_surprisal_bits"]
)

# Percentile rank:
# higher score = rarer state
result["anomaly_percentile"] = (
    result["anomaly_score_bits"]
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

    if percentile >= 99:
        return "EXTREME"

    if percentile >= 95:
        return "HIGH"

    if percentile >= 90:
        return "ELEVATED"

    if percentile >= 75:
        return "MODERATE"

    return "BASELINE"


result["anomaly_tier"] = (
    result["anomaly_percentile"]
    .apply(classify)
)

# ============================================================
# DOMINANT CONTRIBUTOR
# ============================================================

surprisal_columns = [
    f"{f}_surprisal_bits"
    for f in structures
]

result["dominant_contributor"] = (
    result[surprisal_columns]
    .idxmax(axis=1)
    .str.replace(
        "_surprisal_bits",
        "",
        regex=False
    )
)

result["dominant_contribution_bits"] = (
    result[surprisal_columns]
    .max(axis=1)
)

result["dominant_contribution_percent"] = (
    result["dominant_contribution_bits"]
    /
    result["anomaly_score_bits"]
    * 100
)

# ============================================================
# EMPIRICAL STATE VECTOR
# ============================================================

result["occupant_state_vector"] = (
    result[features]
    .astype(str)
    .agg("|".join, axis=1)
)

# ============================================================
# SAVE COMPLETE ANOMALY DATASET
# ============================================================

result.to_csv(
    out / "empirical_occupant_state_anomaly_scores.csv",
    index=False
)

# ============================================================
# ANOMALY SUMMARY
# ============================================================

summary = (
    result["anomaly_tier"]
    .value_counts()
    .rename_axis("anomaly_tier")
    .reset_index(name="records")
)

summary["percentage"] = (
    summary["records"]
    /
    len(result)
    * 100
)

summary.to_csv(
    out / "anomaly_tier_distribution.csv",
    index=False
)

# ============================================================
# TOP ANOMALIES
# ============================================================

top = (
    result
    .sort_values(
        "anomaly_score_bits",
        ascending=False
    )
    .head(100)
)

top.to_csv(
    out / "top_100_empirical_anomalies.csv",
    index=False
)

# ============================================================
# DOMINANT FEATURE SUMMARY
# ============================================================

dominant = (
    result
    .groupby("dominant_contributor")
    .agg(
        records=("anomaly_score_bits", "size"),
        mean_anomaly_score_bits=(
            "anomaly_score_bits",
            "mean"
        ),
        mean_dominant_contribution_percent=(
            "dominant_contribution_percent",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        "mean_dominant_contribution_percent",
        ascending=False
    )
)

dominant.to_csv(
    out / "dominant_anomaly_contributor_summary.csv",
    index=False
)

# ============================================================
# GLOBAL SUMMARY
# ============================================================

global_summary = pd.DataFrame([
    ["observations", len(result)],
    [
        "mean_anomaly_score_bits",
        result["anomaly_score_bits"].mean()
    ],
    [
        "median_anomaly_score_bits",
        result["anomaly_score_bits"].median()
    ],
    [
        "std_anomaly_score_bits",
        result["anomaly_score_bits"].std()
    ],
    [
        "minimum_anomaly_score_bits",
        result["anomaly_score_bits"].min()
    ],
    [
        "maximum_anomaly_score_bits",
        result["anomaly_score_bits"].max()
    ],
    [
        "mean_backoff_level",
        result["total_backoff_level"].mean()
    ],
    [
        "records_extreme",
        (result["anomaly_tier"] == "EXTREME").sum()
    ],
    [
        "records_high",
        (result["anomaly_tier"] == "HIGH").sum()
    ],
    [
        "records_elevated",
        (result["anomaly_tier"] == "ELEVATED").sum()
    ]
], columns=["metric", "value"])

global_summary.to_csv(
    out / "phase6_anomaly_summary.csv",
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("PHASE 6 — EMPIRICAL OCCUPANT STATE ANOMALY")
print("==============================================")

print(
    "\nGLOBAL SUMMARY:"
)

print(
    global_summary.to_string(index=False)
)

print(
    "\nANOMALY TIERS:"
)

print(
    summary.to_string(index=False)
)

print(
    "\nDOMINANT CONTRIBUTORS:"
)

print(
    dominant.to_string(index=False)
)

print(
    "\nTOP 20 ANOMALIES:"
)

print(
    top[
        [
            "anomaly_score_bits",
            "anomaly_percentile",
            "anomaly_tier",
            "dominant_contributor",
            "dominant_contribution_bits",
            "dominant_contribution_percent",
            "seat_position",
            "occupant_role",
            "seat_track_position",
            "posture",
            "seat_type",
            "seat_performance",
            "belt_use",
            "belt_failure",
            "belt_anchorage"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print(
    "\nFILES CREATED:"
)

for f in [
    "empirical_occupant_state_anomaly_scores.csv",
    "anomaly_tier_distribution.csv",
    "top_100_empirical_anomalies.csv",
    "dominant_anomaly_contributor_summary.csv",
    "phase6_anomaly_summary.csv"
]:

    print(out / f)

