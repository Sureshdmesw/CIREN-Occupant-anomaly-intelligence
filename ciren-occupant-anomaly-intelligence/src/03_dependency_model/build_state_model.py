import pandas as pd
from pathlib import Path

src = Path(r"D:\Datasets\occupant_position_ground_truth\ciren_occupant_normalized.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(src)

state = pd.DataFrame({
    "vehicle_id": df["vehicle_id"],
    "occupant_id": df["occupant_id"],
    "seat_location_id": df["seat_location_id"],

    "seat_position": df["seat_position"],
    "occupant_role": df["occupant_role"],
    "seat_track_position": df["seat_track_position"],
    "posture": df["posture"],
    "seat_type": df["seat_type"],
    "seat_performance": df["seat_performance"],

    "belt_availability": df["belt_availability"],
    "belt_use": df["belt_use"],
    "belt_failure": df["belt_failure"],
    "belt_anchorage": df["belt_anchorage"]
})

state.to_csv(
    out / "ciren_empirical_occupant_state.csv",
    index=False
)

print("CREATED:", out / "ciren_empirical_occupant_state.csv")
print("ROWS:", len(state))
print("STATE VARIABLES:", len(state.columns) - 3)
