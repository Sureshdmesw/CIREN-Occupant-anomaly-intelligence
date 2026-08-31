import pandas as pd
from pathlib import Path

base = Path(r"D:\Datasets\occupant_position_ground_truth")

scored_file = (
    base /
    "phase9_anomaly_scoring" /
    "scored_occupant_events_10000.csv"
)

explain_file = (
    base /
    "phase9_explainability" /
    "phase9E_explainable_events.csv"
)

vehicle_file = (
    base /
    "phase9_vehicle_aggregation" /
    "phase9F_vehicle_anomaly_events.csv"
)

out = base / "phase9_splunk_ready"
out.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

scored = pd.read_csv(scored_file)
explain = pd.read_csv(explain_file)
vehicle = pd.read_csv(vehicle_file)

print("SCORED EVENTS:", len(scored))
print("EXPLAINABLE EVENTS:", len(explain))
print("VEHICLE EVENTS:", len(vehicle))

# ============================================================
# VERIFY EVENT IDENTIFIERS
# ============================================================

if "event_id" not in scored.columns:
    raise ValueError("scored dataset missing event_id")

if "event_id" not in explain.columns:
    raise ValueError("explainability dataset missing event_id")

if "vehicle_id" not in scored.columns:
    raise ValueError("scored dataset missing vehicle_id")

if "vehicle_id" not in vehicle.columns:
    raise ValueError("vehicle dataset missing vehicle_id")

# ============================================================
# SELECT EXPLAINABILITY FIELDS
# ============================================================

explain_cols = [
    "event_id",
    "total_surprisal_bits",
    "occupant_anomaly_percentile",
    "anomaly_tier",
    "anomaly_severity",
    "dominant_contributor",
    "dominant_contribution_percent",
    "top_3_contributors",
    "anomaly_signature"
]

missing = [
    c for c in explain_cols
    if c not in explain.columns
]

if missing:
    raise ValueError(
        "Missing explainability fields: " +
        str(missing)
    )

explain_selected = explain[explain_cols].copy()

# ============================================================
# MERGE EXPLAINABILITY
# ============================================================

canonical = scored.merge(
    explain_selected,
    on="event_id",
    how="left",
    validate="one_to_one",
    suffixes=("", "_explain")
)

# ============================================================
# VERIFY EXPLAINABILITY COVERAGE
# ============================================================

required_anomaly = [
    "total_surprisal_bits",
    "occupant_anomaly_percentile",
    "anomaly_tier",
    "anomaly_severity",
    "dominant_contributor",
    "dominant_contribution_percent"
]

missing_anomaly = [
    c for c in required_anomaly
    if c not in canonical.columns
]

if missing_anomaly:
    raise ValueError(
        "Missing anomaly fields after merge: " +
        str(missing_anomaly)
    )

coverage = canonical[required_anomaly].notna().all(axis=1).mean() * 100

print("ANOMALY FIELD COVERAGE:", round(coverage, 3), "%")

if coverage < 100:
    raise ValueError(
        "Not all events have anomaly fields."
    )

# ============================================================
# MERGE VEHICLE-LEVEL CONTEXT
# ============================================================

vehicle_cols = [
    "vehicle_id",
    "vehicle_total_surprisal_bits",
    "vehicle_mean_surprisal_bits",
    "vehicle_median_surprisal_bits",
    "vehicle_max_surprisal_bits",
    "vehicle_anomaly_severity",
    "critical_occupant_count",
    "high_occupant_count",
    "elevated_occupant_count",
    "moderate_occupant_count",
    "low_occupant_count",
    "dominant_vehicle_contributor",
    "dominant_vehicle_contributor_count",
    "dominant_vehicle_contributor_percent",
    "max_anomaly_event_id",
    "max_anomaly_occupant_id",
    "max_anomaly_tier",
    "max_anomaly_signature",
    "vehicle_anomaly_index",
    "vehicle_anomaly_signature"
]

missing_vehicle = [
    c for c in vehicle_cols
    if c not in vehicle.columns
]

if missing_vehicle:
    raise ValueError(
        "Missing vehicle fields: " +
        str(missing_vehicle)
    )

vehicle_selected = vehicle[vehicle_cols].copy()

canonical = canonical.merge(
    vehicle_selected,
    on="vehicle_id",
    how="left",
    validate="many_to_one"
)

# ============================================================
# ADD SPLUNK METADATA
# ============================================================

canonical["splunk_sourcetype"] = "occupant_state"
canonical["splunk_source"] = "ciren_occupant_model"
canonical["splunk_index"] = "occupant_analytics"

canonical["provenance"] = (
    "CIREN_EMPIRICAL_MODEL_SYNTHETIC_RUNTIME_EVENT"
)

canonical["geometry_status"] = (
    canonical["geometry_status"]
)

canonical["temporal_status"] = (
    canonical["temporal_status"]
)

# ============================================================
# EVENT ID UNIQUENESS
# ============================================================

duplicate_events = canonical["event_id"].duplicated().sum()

if duplicate_events:
    raise ValueError(
        f"Duplicate event IDs detected: {duplicate_events}"
    )

# ============================================================
# FINAL COLUMN ORDER
# ============================================================

identity = [
    "event_time",
    "event_id",
    "vehicle_id",
    "occupant_id",
    "seat_location_id"
]

state = [
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

probability = [
    c for c in canonical.columns
    if c.endswith("_probability")
]

surprisal = [
    c for c in canonical.columns
    if c.endswith("_surprisal_bits")
]

anomaly = [
    "total_surprisal_bits",
    "occupant_anomaly_score_bits",
    "occupant_anomaly_percentile",
    "anomaly_tier",
    "anomaly_severity",
    "dominant_contributor",
    "dominant_contribution_percent",
    "top_3_contributors",
    "anomaly_signature"
]

vehicle_state = [
    c for c in vehicle_cols
    if c != "vehicle_id"
]

metadata = [
    "source_type",
    "source_reference",
    "model_type",
    "model_version",
    "geometry_status",
    "temporal_status",
    "splunk_sourcetype",
    "splunk_source",
    "splunk_index",
    "provenance"
]

ordered = []

for group in [
    identity,
    state,
    probability,
    surprisal,
    anomaly,
    vehicle_state,
    metadata
]:
    for c in group:
        if c in canonical.columns and c not in ordered:
            ordered.append(c)

# Include anything unexpected at the end rather than silently dropping it.
for c in canonical.columns:
    if c not in ordered:
        ordered.append(c)

canonical = canonical[ordered]

# ============================================================
# WRITE CSV
# ============================================================

csv_file = (
    out /
    "splunk_canonical_occupant_events.csv"
)

canonical.to_csv(
    csv_file,
    index=False
)

# ============================================================
# WRITE JSON LINES
# ============================================================

json_file = (
    out /
    "splunk_canonical_occupant_events.json"
)

canonical.to_json(
    json_file,
    orient="records",
    lines=True
)

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([
    ["events", len(canonical)],
    ["unique_event_ids", canonical["event_id"].nunique()],
    ["unique_vehicles", canonical["vehicle_id"].nunique()],
    ["unique_occupants", canonical["occupant_id"].nunique()],
    ["duplicate_event_ids", duplicate_events],
    ["anomaly_coverage_percent", coverage],
    [
        "critical_events",
        (canonical["anomaly_severity"] == "CRITICAL").sum()
    ],
    [
        "high_events",
        (canonical["anomaly_severity"] == "HIGH").sum()
    ],
    [
        "elevated_events",
        (canonical["anomaly_severity"] == "ELEVATED").sum()
    ],
    [
        "moderate_events",
        (canonical["anomaly_severity"] == "MODERATE").sum()
    ],
    [
        "low_events",
        (canonical["anomaly_severity"] == "LOW").sum()
    ]
], columns=["metric", "value"])

summary.to_csv(
    out / "phase9H_splunk_stream_summary.csv",
    index=False
)

print("\n==============================================")
print("PHASE 9H — SPLUNK CANONICAL EVENT STREAM")
print("==============================================")

print("\nROWS:", len(canonical))
print("COLUMNS:", len(canonical.columns))
print("UNIQUE EVENTS:", canonical["event_id"].nunique())
print("UNIQUE VEHICLES:", canonical["vehicle_id"].nunique())
print("UNIQUE OCCUPANTS:", canonical["occupant_id"].nunique())
print("DUPLICATE EVENTS:", duplicate_events)

print("\nANOMALY SEVERITY:")
print(
    canonical["anomaly_severity"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nVEHICLE SEVERITY:")
print(
    canonical["vehicle_anomaly_severity"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nFILES CREATED:")
print(csv_file)
print(json_file)
print(out / "phase9H_splunk_stream_summary.csv")

