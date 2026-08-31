import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PHASE 9E — ANOMALY EXPLAINABILITY
# ============================================================

root = Path(
    r"D:\Datasets\occupant_position_ground_truth"
)

src = (
    root /
    "phase9_anomaly_scoring" /
    "scored_occupant_events_10000.csv"
)

out = (
    root /
    "phase9_explainability"
)

df = pd.read_csv(src)

# ------------------------------------------------------------
# Surprisal columns
# ------------------------------------------------------------

score_cols = [
    c for c in df.columns
    if c.endswith("_surprisal_bits")
]

features = [
    c.replace(
        "_surprisal_bits",
        ""
    )
    for c in score_cols
]

# ------------------------------------------------------------
# Total score
# ------------------------------------------------------------

df["total_surprisal_bits"] = (
    df[score_cols]
    .sum(axis=1)
)

# ------------------------------------------------------------
# Contribution percentage
# ------------------------------------------------------------

for score_col, feature in zip(
    score_cols,
    features
):

    contribution_col = (
        feature +
        "_contribution_percent"
    )

    df[contribution_col] = np.where(
        df["total_surprisal_bits"] > 0,
        (
            df[score_col]
            /
            df["total_surprisal_bits"]
            * 100
        ),
        0
    )

# ------------------------------------------------------------
# Top contributor
# ------------------------------------------------------------

contribution_cols = [
    c for c in df.columns
    if c.endswith(
        "_contribution_percent"
    )
]

contribution_matrix = df[
    contribution_cols
].copy()

contribution_matrix.columns = [
    c.replace(
        "_contribution_percent",
        ""
    )
    for c in contribution_cols
]

df["dominant_contributor"] = (
    contribution_matrix
    .idxmax(axis=1)
)

df["dominant_contribution_percent"] = (
    contribution_matrix.max(axis=1)
)

# ------------------------------------------------------------
# Top 3 contributors
# ------------------------------------------------------------

def top_contributors(row):

    values = (
        row.sort_values(
            ascending=False
        )
        .head(3)
    )

    return " | ".join(
        [
            f"{feature}:{value:.2f}%"
            for feature, value
            in values.items()
            if value > 0
        ]
    )

df["top_3_contributors"] = (
    contribution_matrix
    .apply(
        top_contributors,
        axis=1
    )
)

# ------------------------------------------------------------
# Human-readable anomaly signature
# ------------------------------------------------------------

def build_signature(row):

    tier = row["anomaly_tier"]

    dominant = row[
        "dominant_contributor"
    ]

    contribution = row[
        "dominant_contribution_percent"
    ]

    return (
        f"{tier}: "
        f"{dominant.replace('_', ' ')} "
        f"dominates anomaly "
        f"({contribution:.1f}% contribution)"
    )

df["anomaly_signature"] = (
    df.apply(
        build_signature,
        axis=1
    )
)

# ------------------------------------------------------------
# Anomaly severity score
# ------------------------------------------------------------

def severity(score):

    if score < 8:
        return "LOW"

    if score < 12:
        return "MODERATE"

    if score < 16:
        return "ELEVATED"

    if score < 20:
        return "HIGH"

    return "CRITICAL"

df["anomaly_severity"] = (
    df["total_surprisal_bits"]
    .apply(severity)
)

# ------------------------------------------------------------
# Compact explainability dataset
# ------------------------------------------------------------

explain_cols = [
    "event_time",
    "event_id",
    "vehicle_id",
    "occupant_id",
    "seat_position",
    "occupant_role",
    "posture",
    "seat_track_position",
    "seat_type",
    "seat_performance",
    "belt_use",
    "belt_failure",
    "belt_anchorage",
    "total_surprisal_bits",
    "occupant_anomaly_percentile",
    "anomaly_tier",
    "anomaly_severity",
    "dominant_contributor",
    "dominant_contribution_percent",
    "top_3_contributors",
    "anomaly_signature",
    "source_type",
    "source_reference",
    "model_type",
    "model_version"
]

explain = df[
    explain_cols
].copy()

# ------------------------------------------------------------
# Contributor ranking
# ------------------------------------------------------------

rows = []

for feature in features:

    contribution = (
        df[
            feature +
            "_contribution_percent"
        ]
        .mean()
    )

    rows.append({
        "feature": feature,
        "mean_contribution_percent":
            contribution,
        "dominant_count":
            (
                df[
                    "dominant_contributor"
                ] == feature
            ).sum()
    })

ranking = pd.DataFrame(rows)

ranking = ranking.sort_values(
    "mean_contribution_percent",
    ascending=False
)

# ------------------------------------------------------------
# Signature frequency
# ------------------------------------------------------------

signatures = (
    df[
        "anomaly_signature"
    ]
    .value_counts()
    .rename_axis(
        "anomaly_signature"
    )
    .reset_index(
        name="event_count"
    )
)

signatures[
    "event_percentage"
] = (
    signatures["event_count"]
    / len(df)
    * 100
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

event_file = (
    out /
    "phase9E_explainable_events.csv"
)

ranking_file = (
    out /
    "phase9E_contributor_ranking.csv"
)

signature_file = (
    out /
    "phase9E_anomaly_signatures.csv"
)

explain.to_csv(
    event_file,
    index=False
)

ranking.to_csv(
    ranking_file,
    index=False
)

signatures.to_csv(
    signature_file,
    index=False
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\n==============================================")
print("PHASE 9E — ANOMALY EXPLAINABILITY")
print("==============================================")

print(
    "\nEVENTS:",
    len(explain)
)

print(
    "\nTOP CONTRIBUTORS:"
)

print(
    ranking.to_string(
        index=False
    )
)

print(
    "\nTOP ANOMALY SIGNATURES:"
)

print(
    signatures.head(20)
    .to_string(index=False)
)

print(
    "\nSEVERITY DISTRIBUTION:"
)

print(
    explain[
        "anomaly_severity"
    ]
    .value_counts()
    .to_string()
)

print(
    "\nFILES CREATED:"
)

print(event_file)
print(ranking_file)
print(signature_file)

