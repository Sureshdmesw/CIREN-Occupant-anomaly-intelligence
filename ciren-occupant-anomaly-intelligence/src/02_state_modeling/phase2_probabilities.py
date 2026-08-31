import pandas as pd
from pathlib import Path

src = Path(r"D:\Datasets\occupant_position_ground_truth\ciren_occupant_normalized.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(src)

def conditional_probability(a, b, filename):
    table = pd.crosstab(df[a], df[b], normalize="index") * 100
    table.round(3).to_csv(out / filename)

conditional_probability(
    "seat_position",
    "occupant_role",
    "P_role_given_seat_position.csv"
)

conditional_probability(
    "seat_position",
    "posture",
    "P_posture_given_seat_position.csv"
)

conditional_probability(
    "seat_position",
    "belt_use",
    "P_belt_use_given_seat_position.csv"
)

conditional_probability(
    "occupant_role",
    "seat_track_position",
    "P_seat_track_given_role.csv"
)

conditional_probability(
    "occupant_role",
    "posture",
    "P_posture_given_role.csv"
)

print("Conditional-probability matrices created.")
