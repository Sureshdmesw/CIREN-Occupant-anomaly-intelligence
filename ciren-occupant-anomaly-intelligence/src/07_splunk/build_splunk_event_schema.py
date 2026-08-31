import pandas as pd
from pathlib import Path

out = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase9_event_schema"
)

# ============================================================
# SPLUNK OCCUPANT-STATE EVENT SCHEMA
# ============================================================

schema = [

    # -----------------------------
    # Event identity
    # -----------------------------

    ["event_time",
     "string",
     "telemetry_runtime",
     "Event timestamp supplied by telemetry/event generator"],

    ["event_id",
     "string",
     "derived",
     "Unique event identifier"],

    ["vehicle_id",
     "string",
     "CIREN",
     "Vehicle identifier"],

    ["occupant_id",
     "string",
     "CIREN",
     "Occupant identifier"],

    ["seat_location_id",
     "string",
     "CIREN",
     "Seat location identifier"],

    # -----------------------------
    # Occupant state
    # -----------------------------

    ["seat_position",
     "categorical",
     "CIREN",
     "Observed seat position"],

    ["occupant_role",
     "categorical",
     "CIREN",
     "Driver/passenger role"],

    ["seat_track_position",
     "categorical",
     "CIREN",
     "Observed seat-track state"],

    ["posture",
     "categorical",
     "CIREN",
     "Observed occupant posture"],

    ["seat_type",
     "categorical",
     "CIREN",
     "Observed seat type"],

    ["seat_performance",
     "categorical",
     "CIREN",
     "Observed seat performance state"],

    # -----------------------------
    # Restraint state
    # -----------------------------

    ["belt_availability",
     "categorical",
     "CIREN",
     "Observed belt availability"],

    ["belt_use",
     "categorical",
     "CIREN",
     "Observed belt use"],

    ["belt_failure",
     "categorical",
     "CIREN",
     "Observed belt failure state"],

    ["belt_anchorage",
     "categorical",
     "CIREN",
     "Observed belt anchorage state"],

    # -----------------------------
    # Analytical outputs
    # -----------------------------

    ["occupant_anomaly_score_bits",
     "float",
     "derived",
     "Negative log likelihood / surprisal of occupant state"],

    ["occupant_anomaly_percentile",
     "float",
     "derived",
     "Empirical percentile of occupant anomaly score"],

    ["anomaly_tier",
     "categorical",
     "derived",
     "Baseline/moderate/elevated/high/extreme"],

    ["dominant_contributor",
     "categorical",
     "derived",
     "State dimension contributing most strongly to anomaly"],

    ["vehicle_anomaly_score_bits",
     "float",
     "derived",
     "Vehicle-level maximum occupant anomaly"],

    ["vehicle_anomaly_tier",
     "categorical",
     "derived",
     "Vehicle-level anomaly classification"],

    # -----------------------------
    # Explainability
    # -----------------------------

    ["model_type",
     "string",
     "derived",
     "Dependency-aware empirical occupant-state model"],

    ["model_version",
     "string",
     "derived",
     "Version identifier for analytical model"],

    ["source_type",
     "string",
     "metadata",
     "CIREN observed / telemetry generated / simulated"],

    ["source_reference",
     "string",
     "metadata",
     "Dataset or telemetry source"],

    ["geometry_status",
     "string",
     "metadata",
     "Whether geometric occupant coordinates are available"],

    ["temporal_status",
     "string",
     "metadata",
     "Whether event has valid temporal telemetry ordering"]
]

schema_df = pd.DataFrame(
    schema,
    columns=[
        "field",
        "data_type",
        "provenance_type",
        "description"
    ]
)

schema_df.to_csv(
    out / "splunk_occupant_event_schema.csv",
    index=False
)

# ============================================================
# FIELD GROUPS
# ============================================================

groups = pd.DataFrame([

    ["event_identity",
     "event_time,event_id,vehicle_id,occupant_id,seat_location_id"],

    ["occupant_state",
     "seat_position,occupant_role,seat_track_position,posture,seat_type,seat_performance"],

    ["restraint_state",
     "belt_availability,belt_use,belt_failure,belt_anchorage"],

    ["occupant_anomaly",
     "occupant_anomaly_score_bits,occupant_anomaly_percentile,anomaly_tier,dominant_contributor"],

    ["vehicle_anomaly",
     "vehicle_anomaly_score_bits,vehicle_anomaly_tier"],

    ["explainability",
     "model_type,model_version,source_type,source_reference,geometry_status,temporal_status"]

],
columns=[
    "field_group",
    "fields"
])

groups.to_csv(
    out / "splunk_event_field_groups.csv",
    index=False
)

# ============================================================
# DATA PROVENANCE RULES
# ============================================================

rules = pd.DataFrame([

    [
        "CIREN_OBSERVED",
        "Directly observed/reported CIREN occupant-state variable"
    ],

    [
        "MODEL_DERIVED",
        "Calculated from the empirical dependency model"
    ],

    [
        "TELEMETRY_RUNTIME",
        "Supplied by future telemetry/event-generation layer"
    ],

    [
        "NOT_CIREN",
        "Must not be represented as directly observed CIREN data"
    ],

    [
        "TEMPORAL_LIMITATION",
        "CIREN OA contains no timestamp or valid temporal ordering"
    ],

    [
        "GEOMETRY_LIMITATION",
        "CIREN OA does not provide continuous H-point coordinates or live occupant geometry"
    ]

],
columns=[
    "rule",
    "description"
])

rules.to_csv(
    out / "splunk_event_provenance_rules.csv",
    index=False
)

print("\n==============================================")
print("PHASE 9B — SPLUNK EVENT SCHEMA")
print("==============================================")

print("\nFIELDS:", len(schema_df))

print("\nFIELD GROUPS:")
print(
    groups.to_string(
        index=False
    )
)

print("\nPROVENANCE RULES:")
print(
    rules.to_string(
        index=False
    )
)

print("\nFILES CREATED:")

for f in [
    "splunk_occupant_event_schema.csv",
    "splunk_event_field_groups.csv",
    "splunk_event_provenance_rules.csv"
]:
    print(out / f)

