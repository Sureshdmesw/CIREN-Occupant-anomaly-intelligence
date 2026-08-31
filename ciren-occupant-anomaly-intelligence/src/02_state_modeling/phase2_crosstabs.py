import pandas as pd
from pathlib import Path

src = Path(r"D:\Datasets\occupant_position_ground_truth\ciren_occupant_normalized.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(src)

tables = {
    "seat_position_by_role":
        pd.crosstab(df["seat_position"], df["occupant_role"]),

    "posture_by_seat_position":
        pd.crosstab(df["seat_position"], df["posture"]),

    "belt_use_by_seat_position":
        pd.crosstab(df["seat_position"], df["belt_use"]),

    "seat_track_by_role":
        pd.crosstab(df["occupant_role"], df["seat_track_position"]),

    "posture_by_role":
        pd.crosstab(df["occupant_role"], df["posture"]),

    "seat_type_by_seat_position":
        pd.crosstab(df["seat_position"], df["seat_type"]),

    "seat_performance_by_seat_position":
        pd.crosstab(df["seat_position"], df["seat_performance"])
}

for name, table in tables.items():
    table.to_csv(out / f"{name}.csv")

print("Cross-tabulations created:", len(tables))
