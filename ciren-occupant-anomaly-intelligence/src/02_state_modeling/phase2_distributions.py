import pandas as pd
from pathlib import Path

src = Path(r"D:\Datasets\occupant_position_ground_truth\ciren_occupant_normalized.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(src)

reports = {
    "seat_position": "seat_position",
    "occupant_role": "occupant_role",
    "seat_track_position": "seat_track_position",
    "posture": "posture",
    "seat_type": "seat_type",
    "seat_performance": "seat_performance",
    "belt_availability": "belt_availability",
    "belt_use": "belt_use",
    "belt_failure": "belt_failure",
    "belt_anchorage": "belt_anchorage"
}

for name, col in reports.items():
    r = (
        df[col]
        .value_counts(dropna=False)
        .rename_axis(col)
        .reset_index(name="count")
    )
    r["percentage"] = (r["count"] / len(df) * 100).round(3)
    r.to_csv(out / f"{name}_distribution.csv", index=False)

print("Distribution reports created:", len(reports))
