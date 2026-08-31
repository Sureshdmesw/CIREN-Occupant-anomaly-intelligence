import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PHASE 9F — VEHICLE-LEVEL ANOMALY AGGREGATION
# ============================================================

root = Path(
    r"D:\Datasets\occupant_position_ground_truth"
)

src = (
    root /
    "phase9_explainability" /
    "phase9E_explainable_events.csv"
)

out = (
    root /
    "phase9_vehicle_aggregation"
)

df = pd.read_csv(src)

# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

required = [
    "vehicle_id",
    "occupant_id",
    "total_surprisal_bits",
    "anomaly_tier",
    "anomaly_severity",
    "dominant_contributor",
    "dominant_contribution_percent"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ------------------------------------------------------------
# Severity ordering
# ------------------------------------------------------------

severity_order = {
    "LOW": 1,
    "MODERATE": 2,
    "ELEVATED": 3,
    "HIGH": 4,
    "CRITICAL": 5
}

df["severity_numeric"] = (
    df["anomaly_severity"]
    .map(severity_order)
    .fillna(0)
)

# ------------------------------------------------------------
# Vehicle aggregation
# ------------------------------------------------------------

vehicle_rows = []

for vehicle_id, g in df.groupby(
    "vehicle_id",
    dropna=False
):

    occupant_count = len(g)

    total_score = (
        g["total_surprisal_bits"]
        .sum()
    )

    mean_score = (
        g["total_surprisal_bits"]
        .mean()
    )

    max_score = (
        g["total_surprisal_bits"]
        .max()
    )

    median_score = (
        g["total_surprisal_bits"]
        .median()
    )

    max_row = g.loc[
        g["total_surprisal_bits"].idxmax()
    ]

    critical_count = (
        g["anomaly_severity"]
        == "CRITICAL"
    ).sum()

    high_count = (
        g["anomaly_severity"]
        == "HIGH"
    ).sum()

    elevated_count = (
        g["anomaly_severity"]
        == "ELEVATED"
    ).sum()

    moderate_count = (
        g["anomaly_severity"]
        == "MODERATE"
    ).sum()

    low_count = (
        g["anomaly_severity"]
        == "LOW"
    ).sum()

    # Maximum severity present in vehicle
    vehicle_severity = (
        g["severity_numeric"]
        .max()
    )

    reverse_severity = {
        1: "LOW",
        2: "MODERATE",
        3: "ELEVATED",
        4: "HIGH",
        5: "CRITICAL"
    }

    vehicle_severity_label = (
        reverse_severity
        .get(
            int(vehicle_severity),
            "UNKNOWN"
        )
    )

    # Most frequent dominant contributor
    contributor = (
        g["dominant_contributor"]
        .value_counts()
        .idxmax()
    )

    contributor_count = (
        g["dominant_contributor"]
        .value_counts()
        .max()
    )

    contributor_percent = (
        contributor_count /
        occupant_count *
        100
    )

    vehicle_rows.append({

        "vehicle_id":
            vehicle_id,

        "occupant_event_count":
            occupant_count,

        "vehicle_total_surprisal_bits":
            total_score,

        "vehicle_mean_surprisal_bits":
            mean_score,

        "vehicle_median_surprisal_bits":
            median_score,

        "vehicle_max_surprisal_bits":
            max_score,

        "vehicle_anomaly_severity":
            vehicle_severity_label,

        "critical_occupant_count":
            critical_count,

        "high_occupant_count":
            high_count,

        "elevated_occupant_count":
            elevated_count,

        "moderate_occupant_count":
            moderate_count,

        "low_occupant_count":
            low_count,

        "dominant_vehicle_contributor":
            contributor,

        "dominant_vehicle_contributor_count":
            contributor_count,

        "dominant_vehicle_contributor_percent":
            contributor_percent,

        "max_anomaly_event_id":
            max_row["event_id"],

        "max_anomaly_occupant_id":
            max_row["occupant_id"],

        "max_anomaly_tier":
            max_row["anomaly_tier"],

        "max_anomaly_signature":
            max_row["anomaly_signature"],

        "source_type":
            max_row["source_type"],

        "source_reference":
            max_row["source_reference"],

        "model_type":
            max_row["model_type"],

        "model_version":
            max_row["model_version"]

    })

vehicles = pd.DataFrame(
    vehicle_rows
)

# ------------------------------------------------------------
# Vehicle anomaly index
# ------------------------------------------------------------

# Normalized maximum occupant anomaly
score_min = (
    vehicles["vehicle_max_surprisal_bits"]
    .min()
)

score_max = (
    vehicles["vehicle_max_surprisal_bits"]
    .max()
)

if score_max > score_min:

    vehicles[
        "vehicle_anomaly_index"
    ] = (
        (
            vehicles[
                "vehicle_max_surprisal_bits"
            ]
            - score_min
        )
        /
        (
            score_max - score_min
        )
        * 100
    )

else:

    vehicles[
        "vehicle_anomaly_index"
    ] = 0.0

# ------------------------------------------------------------
# Vehicle anomaly signature
# ------------------------------------------------------------

def vehicle_signature(row):

    return (
        f"{row['vehicle_anomaly_severity']}: "
        f"{row['dominant_vehicle_contributor'].replace('_', ' ')} "
        f"dominates vehicle anomaly; "
        f"{int(row['occupant_event_count'])} occupant event(s), "
        f"maximum surprisal "
        f"{row['vehicle_max_surprisal_bits']:.2f} bits"
    )

vehicles[
    "vehicle_anomaly_signature"
] = vehicles.apply(
    vehicle_signature,
    axis=1
)

# ------------------------------------------------------------
# Save vehicle-level dataset
# ------------------------------------------------------------

vehicle_file = (
    out /
    "phase9F_vehicle_anomaly_events.csv"
)

vehicles.to_csv(
    vehicle_file,
    index=False
)

# ------------------------------------------------------------
# Vehicle severity distribution
# ------------------------------------------------------------

severity_summary = (
    vehicles[
        "vehicle_anomaly_severity"
    ]
    .value_counts()
    .rename_axis(
        "vehicle_anomaly_severity"
    )
    .reset_index(
        name="vehicle_count"
    )
)

severity_summary[
    "vehicle_percentage"
] = (
    severity_summary["vehicle_count"]
    /
    len(vehicles)
    *
    100
)

severity_summary.to_csv(
    out /
    "phase9F_vehicle_severity_summary.csv",
    index=False
)

# ------------------------------------------------------------
# Contributor distribution
# ------------------------------------------------------------

contributor_summary = (
    vehicles[
        "dominant_vehicle_contributor"
    ]
    .value_counts()
    .rename_axis(
        "dominant_vehicle_contributor"
    )
    .reset_index(
        name="vehicle_count"
    )
)

contributor_summary[
    "vehicle_percentage"
] = (
    contributor_summary["vehicle_count"]
    /
    len(vehicles)
    *
    100
)

contributor_summary.to_csv(
    out /
    "phase9F_vehicle_contributor_summary.csv",
    index=False
)

# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------

summary = pd.DataFrame([
    [
        "occupant_events",
        len(df)
    ],
    [
        "unique_vehicles",
        df["vehicle_id"].nunique()
    ],
    [
        "mean_occupants_per_vehicle",
        len(df) /
        df["vehicle_id"].nunique()
    ],
    [
        "maximum_occupants_per_vehicle",
        df.groupby(
            "vehicle_id"
        ).size().max()
    ],
    [
        "mean_vehicle_max_surprisal_bits",
        vehicles[
            "vehicle_max_surprisal_bits"
        ].mean()
    ],
    [
        "maximum_vehicle_surprisal_bits",
        vehicles[
            "vehicle_max_surprisal_bits"
        ].max()
    ]
],
columns=[
    "metric",
    "value"
])

summary.to_csv(
    out /
    "phase9F_vehicle_summary.csv",
    index=False
)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print("\n==============================================")
print("PHASE 9F — VEHICLE-LEVEL ANOMALY AGGREGATION")
print("==============================================")

print(
    "\nOCCUPANT EVENTS:",
    len(df)
)

print(
    "UNIQUE VEHICLES:",
    df["vehicle_id"].nunique()
)

print(
    "\nVEHICLE SUMMARY:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nVEHICLE SEVERITY:"
)

print(
    severity_summary.to_string(
        index=False
    )
)

print(
    "\nDOMINANT VEHICLE CONTRIBUTORS:"
)

print(
    contributor_summary.to_string(
        index=False
    )
)

print(
    "\nTOP 20 VEHICLES BY MAX ANOMALY:"
)

print(
    vehicles
    .sort_values(
        "vehicle_max_surprisal_bits",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

print(
    "\nFILES CREATED:"
)

print(vehicle_file)

print(
    out /
    "phase9F_vehicle_severity_summary.csv"
)

print(
    out /
    "phase9F_vehicle_contributor_summary.csv"
)

print(
    out /
    "phase9F_vehicle_summary.csv"
)

