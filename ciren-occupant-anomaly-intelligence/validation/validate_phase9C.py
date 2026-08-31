import pandas as pd

f = r"D:\Datasets\occupant_position_ground_truth\phase9_event_generation\synthetic_occupant_events_10000.csv"

d = pd.read_csv(f)

cols = [
    "seat_position",
    "occupant_role",
    "seat_track_position",
    "posture",
    "seat_type",
    "belt_use",
    "belt_failure",
    "belt_anchorage"
]

for c in cols:
    print("\n### " + c)
    print(
        d[c]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )
