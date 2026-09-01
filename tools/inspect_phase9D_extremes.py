import pandas as pd

f = r"D:\Datasets\occupant_position_ground_truth\phase9_anomaly_scoring\scored_occupant_events_10000.csv"

d = pd.read_csv(f)

cols = [
    "event_time",
    "event_id",
    "vehicle_id",
    "occupant_id",
    "seat_position",
    "occupant_role",
    "seat_track_position",
    "posture",
    "seat_type",
    "seat_performance",
    "belt_use",
    "belt_failure",
    "belt_anchorage",
    "occupant_anomaly_score_bits",
    "occupant_anomaly_percentile",
    "anomaly_tier",
    "dominant_contributor",
    "dominant_contributor_score_bits"
]

print(
    d[
        cols
    ]
    .sort_values(
        "occupant_anomaly_score_bits",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)
