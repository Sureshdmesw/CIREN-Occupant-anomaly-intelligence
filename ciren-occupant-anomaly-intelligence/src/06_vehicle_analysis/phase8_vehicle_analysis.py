import pandas as pd
import numpy as np
from pathlib import Path

phase6 = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase6_anomaly"
)

phase8 = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase8_case_analysis"
)

src = phase6 / "empirical_occupant_state_anomaly_scores.csv"

df = pd.read_csv(src)

# ============================================================
# REQUIRED IDENTIFIERS
# ============================================================

required = [
    "vehicle_id",
    "occupant_id",
    "anomaly_score_bits",
    "anomaly_tier",
    "dominant_contributor"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ============================================================
# NUMERIC ANOMALY TIER
# ============================================================

tier_order = {
    "BASELINE": 0,
    "MODERATE": 1,
    "ELEVATED": 2,
    "HIGH": 3,
    "EXTREME": 4
}

df["tier_numeric"] = (
    df["anomaly_tier"]
    .map(tier_order)
    .fillna(0)
)

# ============================================================
# VEHICLE-LEVEL AGGREGATION
# ============================================================

vehicle_rows = []

for vehicle_id, g in df.groupby(
    "vehicle_id",
    dropna=False
):

    scores = g["anomaly_score_bits"]

    max_idx = scores.idxmax()

    max_record = g.loc[max_idx]

    tier_counts = (
        g["anomaly_tier"]
        .value_counts()
    )

    dominant_counts = (
        g["dominant_contributor"]
        .value_counts()
    )

    dominant_vehicle_contributor = (
        dominant_counts.index[0]
        if len(dominant_counts) > 0
        else "UNKNOWN"
    )

    vehicle_rows.append({

        "vehicle_id": vehicle_id,

        "occupant_count":
            len(g),

        "mean_anomaly_score_bits":
            scores.mean(),

        "median_anomaly_score_bits":
            scores.median(),

        "maximum_anomaly_score_bits":
            scores.max(),

        "std_anomaly_score_bits":
            scores.std()
            if len(g) > 1
            else 0.0,

        "minimum_anomaly_score_bits":
            scores.min(),

        "baseline_occupants":
            tier_counts.get(
                "BASELINE",
                0
            ),

        "moderate_occupants":
            tier_counts.get(
                "MODERATE",
                0
            ),

        "elevated_occupants":
            tier_counts.get(
                "ELEVATED",
                0
            ),

        "high_occupants":
            tier_counts.get(
                "HIGH",
                0
            ),

        "extreme_occupants":
            tier_counts.get(
                "EXTREME",
                0
            ),

        "anomalous_occupants":
            (
                g["anomaly_tier"]
                != "BASELINE"
            ).sum(),

        "high_or_extreme_occupants":
            (
                g["tier_numeric"] >= 3
            ).sum(),

        "extreme_occupant_count":
            (
                g["tier_numeric"] == 4
            ).sum(),

        "anomalous_occupant_fraction":
            (
                (g["anomaly_tier"] != "BASELINE")
                .mean()
            ),

        "high_or_extreme_fraction":
            (
                (g["tier_numeric"] >= 3)
                .mean()
            ),

        "maximum_anomaly_occupant_id":
            max_record["occupant_id"],

        "maximum_anomaly_dominant_contributor":
            max_record["dominant_contributor"],

        "maximum_anomaly_tier":
            max_record["anomaly_tier"],

        "vehicle_dominant_contributor":
            dominant_vehicle_contributor
    })


vehicle = pd.DataFrame(
    vehicle_rows
)

# ============================================================
# VEHICLE-LEVEL ANOMALY TIER
# ============================================================

def vehicle_tier(row):

    if row["extreme_occupant_count"] >= 1:
        return "EXTREME"

    if row["high_occupants"] >= 1:
        return "HIGH"

    if row["elevated_occupants"] >= 1:
        return "ELEVATED"

    if row["moderate_occupants"] >= 1:
        return "MODERATE"

    return "BASELINE"


vehicle["vehicle_anomaly_tier"] = (
    vehicle.apply(
        vehicle_tier,
        axis=1
    )
)

# ============================================================
# VEHICLE ANOMALY SCORE
# ============================================================

vehicle["vehicle_anomaly_score_bits"] = (
    vehicle["maximum_anomaly_score_bits"]
)

# ============================================================
# MULTI-OCCUPANT INTERACTION FLAG
# ============================================================

vehicle["multi_occupant_anomaly"] = (
    vehicle["anomalous_occupants"] >= 2
)

vehicle["multiple_high_or_extreme"] = (
    vehicle["high_or_extreme_occupants"] >= 2
)

# ============================================================
# RANK VEHICLES
# ============================================================

vehicle = vehicle.sort_values(
    [
        "vehicle_anomaly_score_bits",
        "anomalous_occupants"
    ],
    ascending=False
)

vehicle["vehicle_anomaly_percentile"] = (
    vehicle[
        "vehicle_anomaly_score_bits"
    ]
    .rank(
        pct=True,
        method="average"
    )
    * 100
)

# ============================================================
# SAVE VEHICLE DATASET
# ============================================================

vehicle.to_csv(
    phase8 /
    "vehicle_level_occupant_anomaly.csv",
    index=False
)

# ============================================================
# TIER DISTRIBUTION
# ============================================================

tier_distribution = (
    vehicle["vehicle_anomaly_tier"]
    .value_counts()
    .rename_axis(
        "vehicle_anomaly_tier"
    )
    .reset_index(
        name="vehicles"
    )
)

tier_distribution[
    "percentage"
] = (
    tier_distribution["vehicles"]
    / len(vehicle)
    * 100
)

tier_distribution.to_csv(
    phase8 /
    "vehicle_anomaly_tier_distribution.csv",
    index=False
)

# ============================================================
# CONTRIBUTOR DISTRIBUTION
# ============================================================

contributors = (
    vehicle[
        "vehicle_dominant_contributor"
    ]
    .value_counts()
    .rename_axis(
        "vehicle_dominant_contributor"
    )
    .reset_index(
        name="vehicles"
    )
)

contributors["percentage"] = (
    contributors["vehicles"]
    / len(vehicle)
    * 100
)

contributors.to_csv(
    phase8 /
    "vehicle_dominant_contributor_distribution.csv",
    index=False
)

# ============================================================
# MULTI-OCCUPANT ANALYSIS
# ============================================================

multi_summary = pd.DataFrame([
    [
        "vehicles",
        len(vehicle)
    ],
    [
        "vehicles_with_multiple_occupants",
        (
            vehicle["occupant_count"] > 1
        ).sum()
    ],
    [
        "vehicles_with_multiple_anomalous_occupants",
        (
            vehicle["anomalous_occupants"] >= 2
        ).sum()
    ],
    [
        "vehicles_with_multiple_high_or_extreme",
        (
            vehicle["multiple_high_or_extreme"]
        ).sum()
    ],
    [
        "maximum_occupants_in_vehicle",
        vehicle["occupant_count"].max()
    ]
],
columns=[
    "metric",
    "value"
])

multi_summary.to_csv(
    phase8 /
    "vehicle_multi_occupant_summary.csv",
    index=False
)

# ============================================================
# GLOBAL CASE SUMMARY
# ============================================================

summary = pd.DataFrame([
    [
        "vehicles",
        len(vehicle)
    ],
    [
        "mean_occupants_per_vehicle",
        vehicle["occupant_count"].mean()
    ],
    [
        "maximum_occupants_per_vehicle",
        vehicle["occupant_count"].max()
    ],
    [
        "vehicles_with_anomaly",
        (
            vehicle["anomalous_occupants"] > 0
        ).sum()
    ],
    [
        "vehicles_with_high_or_extreme",
        (
            vehicle["high_or_extreme_occupants"] > 0
        ).sum()
    ],
    [
        "vehicles_with_extreme",
        (
            vehicle["extreme_occupant_count"] > 0
        ).sum()
    ],
    [
        "maximum_vehicle_anomaly_score_bits",
        vehicle[
            "vehicle_anomaly_score_bits"
        ].max()
    ],
    [
        "mean_vehicle_anomaly_score_bits",
        vehicle[
            "vehicle_anomaly_score_bits"
        ].mean()
    ]
],
columns=[
    "metric",
    "value"
])

summary.to_csv(
    phase8 /
    "phase8_vehicle_summary.csv",
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("PHASE 8 — VEHICLE-LEVEL OCCUPANT ANOMALY")
print("==============================================")

print("\nGLOBAL SUMMARY:")
print(
    summary.to_string(
        index=False
    )
)

print("\nVEHICLE TIER DISTRIBUTION:")
print(
    tier_distribution.to_string(
        index=False
    )
)

print("\nDOMINANT CONTRIBUTORS:")
print(
    contributors.to_string(
        index=False
    )
)

print("\nMULTI-OCCUPANT SUMMARY:")
print(
    multi_summary.to_string(
        index=False
    )
)

print("\nTOP 20 VEHICLE ANOMALIES:")
print(
    vehicle[
        [
            "vehicle_id",
            "occupant_count",
            "vehicle_anomaly_score_bits",
            "vehicle_anomaly_percentile",
            "vehicle_anomaly_tier",
            "anomalous_occupants",
            "high_or_extreme_occupants",
            "maximum_anomaly_occupant_id",
            "maximum_anomaly_dominant_contributor",
            "vehicle_dominant_contributor"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nFILES CREATED:")

for f in [
    "vehicle_level_occupant_anomaly.csv",
    "vehicle_anomaly_tier_distribution.csv",
    "vehicle_dominant_contributor_distribution.csv",
    "vehicle_multi_occupant_summary.csv",
    "phase8_vehicle_summary.csv"
]:
    print(
        phase8 / f
    )

