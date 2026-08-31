import pandas as pd
from pathlib import Path

src = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis\ciren_empirical_occupant_state.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(src)

state_cols = [
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

unique_states = (
    df.groupby(state_cols, dropna=False)
      .size()
      .reset_index(name="observations")
      .sort_values("observations", ascending=False)
)

unique_states["probability"] = (
    unique_states["observations"] / len(df)
)

unique_states["probability_percent"] = (
    unique_states["probability"] * 100
).round(4)

unique_states.to_csv(
    out / "empirical_occupant_state_space.csv",
    index=False
)

print("TOTAL OBSERVATIONS:", len(df))
print("DISTINCT OBSERVED STATES:", len(unique_states))
print(
    "STATE SPACE COVERAGE:",
    round(len(unique_states) / len(df) * 100, 2),
    "%"
)

print("\nTOP 20 OBSERVED STATES:")
print(unique_states.head(20).to_string(index=False))
