import pandas as pd
import numpy as np
from pathlib import Path

src = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis\ciren_empirical_occupant_state.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(src)

pairs = [
    ("seat_position", "occupant_role"),
    ("seat_position", "seat_track_position"),
    ("seat_position", "posture"),
    ("seat_position", "seat_type"),
    ("seat_position", "belt_use"),
    ("occupant_role", "seat_track_position"),
    ("occupant_role", "posture"),
    ("occupant_role", "belt_use"),
    ("seat_track_position", "posture"),
    ("seat_track_position", "belt_use"),
    ("posture", "belt_use")
]

for a, b in pairs:

    table = pd.crosstab(
        df[a],
        df[b],
        normalize="index"
    )

    filename = f"conditional_{a}_given_{b}.csv"

    table.round(6).to_csv(out / filename)

    print("\n###", a, "->", b)
    print(table.round(4).to_string())

print("\nCreated", len(pairs), "conditional probability matrices.")
