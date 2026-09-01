import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PHASE 9H — CANONICAL SPLUNK EVENT STREAM
# ============================================================

root = Path(
    r"D:\Datasets\occupant_position_ground_truth"
)

source_file = (
    root /
    "phase9_event_generation" /
    "synthetic_occupant_events_10000.csv"
)

vehicle_file = (
    root /
    "phase9_vehicle_aggregation" /
    "phase9F_vehicle_anomaly_events.csv"
)

out = (
    root /
    "phase9_splunk_ready"
)

occupant = pd.read_csv(source_file)
vehicle = pd.read_csv(vehicle_file)

# ============================================================
# VERIFY REQUIRED SOURCE FIELDS
# ============================================================

required_occupant = [
    "event_time",
    "event_id",
    "vehicle_id",
    "occupant_id",
    "seat_location_id",
    "seat_position",
    "occupant_role",
    "seat_track_position",
    "posture",
    "seat_type",
    "seat_performance",
    "belt_availability",
    "belt_use",
    "belt_failure",
    "belt_anchorage",
    "source_type",
    "source_reference",
    "model_type",
    "model_version",
    "geometry_status",
    "temporal_status"
]

required_vehicle = [
    "vehicle_id",
    "vehicle_anomaly_index",
    "vehicle_anomaly_severity",
    "vehicle_anomaly_signature"
]

missing_occupant = [
    c for c in required_occupant
    if c not in occupant.columns
]

missing_vehicle = [
    c for c in required_vehicle
    if c not in vehicle.columns
]

if missing_occupant:
    raise ValueError(
        "Missing occupant fields: "
        + str(missing_occupant)
    )

if missing_vehicle:
    raise ValueError(
        "Missing vehicle fields: "
        + str(missing_vehicle)
    )

# ============================================================
# OCCUPANT ANOMALY FIELDS
# ============================================================

anomaly_fields = [
    "total_surprisal_bits",
    "anomaly_percentile",
    "anomaly_tier",
    "anomaly_severity",
    "dominant_contributor",
    "dominant_contribution_percent"
]

missing_anomaly = [
    c for c in anomaly_fields
    if c not in occupant.columns
]

if missing_anomaly:
    raise ValueError(
        "Missing anomaly fields: "
        + str(missing_anomaly)
    )

# ============================================================
# MERGE VEHICLE CONTEXT
# ============================================================

vehicle_context = vehicle[
    [
        "vehicle_id",
        "vehicle_anomaly_index",
        "vehicle_anomaly_severity",
        "vehicle_anomaly_signature"
    ]
].copy()

vehicle_context = vehicle_context.rename(
    columns={
        "vehicle_anomaly_severity":
            "vehicle_anomaly_severity",
        "vehicle_anomaly_signature":
            "vehicle_anomaly_signature"
    }
)

events = occupant.merge(
    vehicle_context,
    on="vehicle_id",
    how="left",
    validate="many_to_one"
)

# ============================================================
# SPLUNK FIELD NORMALIZATION
# ============================================================

events["_time"] = events["event_time"]

events["index"] = "occupant_state"

events["sourcetype"] = "occupant:state"

events["source"] = "ciren_empirical_model"

events["host"] = events["vehicle_id"]

events["event_type"] = "occupant_state_event"

events["model_status"] = (
    "EMPIRICAL_MODEL_DERIVED"
)

# ============================================================
# CANONICAL FIELD ORDER
# ============================================================

canonical_fields = [

    "_time",
    "event_time",
    "event_id",
    "event_type",

    "index",
    "sourcetype",
    "source",
    "host",

    "vehicle_id",
    "occupant_id",
    "seat_location_id",

    "seat_position",
    "occupant_role",
    "seat_track_position",
    "posture",
    "seat_type",
    "seat_performance",

    "belt_availability",
    "belt_use",
    "belt_failure",
    "belt_anchorage",

    "total_surprisal_bits",
    "anomaly_percentile",
    "anomaly_tier",
    "anomaly_severity",

    "dominant_contributor",
    "dominant_contribution_percent",

    "vehicle_anomaly_index",
    "vehicle_anomaly_severity",
    "vehicle_anomaly_signature",

    "source_type",
    "source_reference",

    "model_type",
    "model_version",

    "geometry_status",
    "temporal_status",
    "model_status"
]

missing_canonical = [
    c for c in canonical_fields
    if c not in events.columns
]

if missing_canonical:
    raise ValueError(
        "Missing canonical fields: "
        + str(missing_canonical)
    )

events = events[
    canonical_fields
].copy()

# ============================================================
# TYPE NORMALIZATION
# ============================================================

string_fields = [
    "_time",
    "event_time",
    "event_id",
    "event_type",
    "index",
    "sourcetype",
    "source",
    "host",
    "vehicle_id",
    "occupant_id",
    "seat_location_id",
    "seat_position",
    "occupant_role",
    "seat_track_position",
    "posture",
    "seat_type",
    "seat_performance",
    "belt_availability",
    "belt_use",
    "belt_failure",
    "belt_anchorage",
    "anomaly_tier",
    "anomaly_severity",
    "dominant_contributor",
    "vehicle_anomaly_severity",
    "vehicle_anomaly_signature",
    "source_type",
    "source_reference",
    "model_type",
    "model_version",
    "geometry_status",
    "temporal_status",
    "model_status"
]

for c in string_fields:
    events[c] = (
        events[c]
        .fillna("NOT_COLLECTED/UNKNOWN")
        .astype(str)
    )

numeric_fields = [
    "total_surprisal_bits",
    "anomaly_percentile",
    "dominant_contribution_percent",
    "vehicle_anomaly_index"
]

for c in numeric_fields:
    events[c] = pd.to_numeric(
        events[c],
        errors="coerce"
    )

# ============================================================
# EVENT VALIDATION
# ============================================================

if len(events) != len(occupant):
    raise ValueError(
        "Event count changed during vehicle merge."
    )

if events["event_id"].duplicated().any():
    raise ValueError(
        "Duplicate event_id detected."
    )

if events["occupant_id"].duplicated().any():
    raise ValueError(
        "Duplicate occupant_id detected."
    )

if events["vehicle_id"].isna().any():
    raise ValueError(
        "Missing vehicle_id detected."
    )

if events["vehicle_anomaly_severity"].isna().any():
    raise ValueError(
        "Vehicle anomaly merge failed."
    )

# ============================================================
# WRITE CANONICAL CSV
# ============================================================

csv_file = (
    out /
    "splunk_canonical_occupant_events.csv"
)

events.to_csv(
    csv_file,
    index=False
)

# ============================================================
# WRITE SPLUNK JSON EVENT FORMAT
# ============================================================

json_file = (
    out /
    "splunk_canonical_occupant_events.json"
)

events.to_json(
    json_file,
    orient="records",
    lines=True
)

# ============================================================
# FIELD INVENTORY
# ============================================================

field_inventory = pd.DataFrame({

    "field": events.columns,

    "data_type": [
        str(events[c].dtype)
        for c in events.columns
    ],

    "non_null_count": [
        events[c].notna().sum()
        for c in events.columns
    ],

    "unique_values": [
        events[c].nunique()
        for c in events.columns
    ]

})

field_inventory.to_csv(
    out /
    "phase9H_field_inventory.csv",
    index=False
)

# ============================================================
# VALIDATION SUMMARY
# ============================================================

summary = pd.DataFrame([

    [
        "events",
        len(events),
        "Canonical Splunk events"
    ],

    [
        "unique_event_ids",
        events["event_id"].nunique(),
        "Unique event identifiers"
    ],

    [
        "unique_vehicles",
        events["vehicle_id"].nunique(),
        "Unique vehicle identifiers"
    ],

    [
        "unique_occupants",
        events["occupant_id"].nunique(),
        "Unique occupant identifiers"
    ],

    [
        "duplicate_event_ids",
        events["event_id"].duplicated().sum(),
        "Must equal zero"
    ],

    [
        "duplicate_occupant_ids",
        events["occupant_id"].duplicated().sum(),
        "Must equal zero"
    ],

    [
        "missing_vehicle_context",
        events["vehicle_anomaly_severity"].isna().sum(),
        "Must equal zero"
    ],

    [
        "critical_events",
        (
            events["anomaly_severity"]
            == "CRITICAL"
        ).sum(),
        "Critical occupant events"
    ],

    [
        "high_events",
        (
            events["anomaly_severity"]
            == "HIGH"
        ).sum(),
        "High occupant events"
    ]

],
columns=[
    "metric",
    "value",
    "description"
])

summary.to_csv(
    out /
    "phase9H_validation_summary.csv",
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("PHASE 9H — CANONICAL SPLUNK EVENT STREAM")
print("==============================================")

print("\nEVENTS:", len(events))
print(
    "UNIQUE EVENT IDS:",
    events["event_id"].nunique()
)
print(
    "UNIQUE VEHICLES:",
    events["vehicle_id"].nunique()
)
print(
    "UNIQUE OCCUPANTS:",
    events["occupant_id"].nunique()
)

print("\nVALIDATION SUMMARY:")
print(
    summary.to_string(index=False)
)

print("\nFIELD COUNT:", len(events.columns))

print("\nFILES CREATED:")

for f in [
    "splunk_canonical_occupant_events.csv",
    "splunk_canonical_occupant_events.json",
    "phase9H_field_inventory.csv",
    "phase9H_validation_summary.csv"
]:
    print(out / f)

