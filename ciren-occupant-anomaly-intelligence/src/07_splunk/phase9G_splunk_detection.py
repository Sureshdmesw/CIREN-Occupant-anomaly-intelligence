import pandas as pd
from pathlib import Path

# ============================================================
# PHASE 9G — SPLUNK DETECTION SPECIFICATION
# ============================================================

root = Path(
    r"D:\Datasets\occupant_position_ground_truth"
)

vehicle_file = (
    root /
    "phase9_vehicle_aggregation" /
    "phase9F_vehicle_anomaly_events.csv"
)

occupant_file = (
    root /
    "phase9_explainability" /
    "phase9E_explainable_events.csv"
)

out = (
    root /
    "phase9_detection_spec"
)

vehicle = pd.read_csv(vehicle_file)
occupant = pd.read_csv(occupant_file)

# ============================================================
# 1. OCCUPANT DETECTION RULES
# ============================================================

occupant_rules = pd.DataFrame([

    [
        "OCCUPANT_LOW",
        "LOW",
        "LOW",
        "Informational occupant-state deviation",
        "No alert; retain event for baseline analytics"
    ],

    [
        "OCCUPANT_MODERATE",
        "MODERATE",
        "MODERATE",
        "Moderate deviation from empirical occupant-state model",
        "Log and visualize"
    ],

    [
        "OCCUPANT_ELEVATED",
        "ELEVATED",
        "ELEVATED",
        "Elevated occupant-state anomaly",
        "Investigate occupant state"
    ],

    [
        "OCCUPANT_HIGH",
        "HIGH",
        "HIGH",
        "High occupant-state anomaly",
        "Generate Splunk notable event"
    ],

    [
        "OCCUPANT_CRITICAL",
        "EXTREME",
        "CRITICAL",
        "Extreme occupant-state anomaly",
        "Generate high-priority Splunk alert"

    ]

],
columns=[
    "rule_id",
    "anomaly_tier",
    "anomaly_severity",
    "description",
    "recommended_action"
])

occupant_rules.to_csv(
    out /
    "phase9G_splunk_detection_rules.csv",
    index=False
)

# ============================================================
# 2. DATA-DRIVEN THRESHOLDS
# ============================================================

score = occupant[
    "total_surprisal_bits"
].dropna()

threshold_rows = []

for percentile in [50, 75, 90, 95, 99, 99.5]:

    threshold_rows.append({

        "threshold_type":
            "EMPIRICAL_PERCENTILE",

        "percentile":
            percentile,

        "surprisal_bits":
            score.quantile(
                percentile / 100
            ),

        "basis":
            "Synthetic events generated from CIREN empirical dependency model"

    })

thresholds = pd.DataFrame(
    threshold_rows
)

thresholds.to_csv(
    out /
    "phase9G_splunk_alert_thresholds.csv",
    index=False
)

# ============================================================
# 3. VEHICLE CORRELATION RULES
# ============================================================

correlation_rules = pd.DataFrame([

    [
        "VEHICLE_CRITICAL",
        "vehicle_anomaly_severity == CRITICAL",
        "CRITICAL",
        "Immediate vehicle-level anomaly alert"
    ],

    [
        "VEHICLE_HIGH",
        "vehicle_anomaly_severity == HIGH",
        "HIGH",
        "High-priority vehicle anomaly alert"
    ],

    [
        "MULTIPLE_HIGH_OCCUPANTS",
        "high_occupant_count >= 2",
        "HIGH",
        "Multiple high-severity occupants within vehicle"
    ],

    [
        "MULTIPLE_CRITICAL_OCCUPANTS",
        "critical_occupant_count >= 2",
        "CRITICAL",
        "Multiple critical occupants within vehicle"
    ],

    [
        "MULTIPLE_ELEVATED_OCCUPANTS",
        "elevated_occupant_count >= 2",
        "ELEVATED",
        "Multiple elevated occupant anomalies"
    ],

    [
        "DOMINANT_CONTRIBUTOR",
        "dominant_vehicle_contributor_percent >= 50",
        "CONTEXT",
        "Single state dimension dominates vehicle anomaly"
    ]

],
columns=[
    "rule_id",
    "condition",
    "alert_priority",
    "description"
])

correlation_rules.to_csv(
    out /
    "phase9G_splunk_correlation_rules.csv",
    index=False
)

# ============================================================
# 4. OBSERVED DATASET CHARACTERISTICS
# ============================================================

summary = pd.DataFrame([

    [
        "occupant_events",
        len(occupant),
        "Generated occupant events"
    ],

    [
        "vehicles",
        len(vehicle),
        "Vehicle-level aggregated events"
    ],

    [
        "critical_occupants",
        (
            occupant["anomaly_severity"]
            == "CRITICAL"
        ).sum(),
        "Critical occupant events"
    ],

    [
        "high_occupants",
        (
            occupant["anomaly_severity"]
            == "HIGH"
        ).sum(),
        "High occupant events"
    ],

    [
        "critical_vehicles",
        (
            vehicle["vehicle_anomaly_severity"]
            == "CRITICAL"
        ).sum(),
        "Critical vehicle events"
    ],

    [
        "high_vehicles",
        (
            vehicle["vehicle_anomaly_severity"]
            == "HIGH"
        ).sum(),
        "High vehicle events"
    ],

    [
        "maximum_surprisal_bits",
        occupant[
            "total_surprisal_bits"
        ].max(),
        "Maximum generated occupant surprisal"
    ],

    [
        "median_surprisal_bits",
        occupant[
            "total_surprisal_bits"
        ].median(),
        "Median generated occupant surprisal"
    ]

],
columns=[
    "metric",
    "value",
    "description"
])

summary.to_csv(
    out /
    "phase9G_splunk_detection_summary.csv",
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("PHASE 9G — SPLUNK DETECTION SPECIFICATION")
print("==============================================")

print("\nOCCUPANT DETECTION RULES:")
print(
    occupant_rules.to_string(
        index=False
    )
)

print("\nEMPIRICAL THRESHOLDS:")
print(
    thresholds.to_string(
        index=False
    )
)

print("\nVEHICLE CORRELATION RULES:")
print(
    correlation_rules.to_string(
        index=False
    )
)

print("\nDATA SUMMARY:")
print(
    summary.to_string(
        index=False
    )
)

print("\nFILES CREATED:")

for f in [
    "phase9G_splunk_detection_rules.csv",
    "phase9G_splunk_alert_thresholds.csv",
    "phase9G_splunk_correlation_rules.csv",
    "phase9G_splunk_detection_summary.csv"
]:
    print(out / f)

