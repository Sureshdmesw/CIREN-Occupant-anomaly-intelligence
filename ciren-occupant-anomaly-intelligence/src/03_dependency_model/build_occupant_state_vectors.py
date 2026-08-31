import pandas as pd
import numpy as np
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

# Encode every categorical state deterministically.
encoded = df.copy()

encoders = {}

for col in state_cols:
    categories = sorted(encoded[col].fillna("UNKNOWN").astype(str).unique())
    mapping = {value: i for i, value in enumerate(categories)}

    encoded[col + "_code"] = (
        encoded[col]
        .fillna("UNKNOWN")
        .astype(str)
        .map(mapping)
    )

    encoders[col] = mapping

# Construct empirical occupant-state vector.
code_cols = [c + "_code" for c in state_cols]

encoded["state_dimension"] = len(code_cols)

encoded["occupant_state_vector"] = encoded[code_cols].astype(str).agg(
    ",".join,
    axis=1
)

# Frequency of each complete state.
state_frequency = (
    encoded.groupby("occupant_state_vector")
    .size()
    .reset_index(name="observations")
)

state_frequency["empirical_probability"] = (
    state_frequency["observations"] / len(encoded)
)

state_frequency = state_frequency.sort_values(
    "observations",
    ascending=False
)

encoded.to_csv(
    out / "occupant_state_vectors.csv",
    index=False
)

state_frequency.to_csv(
    out / "occupant_state_vector_frequency.csv",
    index=False
)

print("STATE VECTOR RECORDS:", len(encoded))
print("STATE DIMENSIONS:", len(code_cols))
print("DISTINCT STATE VECTORS:", len(state_frequency))

print("\nTOP 15 STATE VECTORS:")
print(state_frequency.head(15).to_string(index=False))

print("\nSTATE VECTOR COLUMNS:")
print(code_cols)
