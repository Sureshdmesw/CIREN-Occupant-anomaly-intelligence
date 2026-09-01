import pandas as pd
import numpy as np
from pathlib import Path

src = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis\ciren_empirical_occupant_state.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(src)

for c in [
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
]:
    df[c] = df[c].fillna("NOT_COLLECTED/UNKNOWN").astype(str)

# ---------------------------------------------------------
# Dependency-aware factorization
# ---------------------------------------------------------

models = {

    "P_seat_position":
        ["seat_position"],

    "P_role_given_seat_position":
        ["seat_position", "occupant_role"],

    "P_track_given_seat_role":
        ["seat_position", "occupant_role", "seat_track_position"],

    "P_type_given_seat_track":
        ["seat_position", "seat_track_position", "seat_type"],

    "P_posture_given_position_track_type":
        ["seat_position", "seat_track_position",
         "seat_type", "posture"],

    "P_belt_use_given_position_role_track_type":
        ["seat_position", "occupant_role",
         "seat_track_position", "seat_type", "belt_use"],

    "P_anchorage_given_type_use":
        ["seat_type", "belt_use", "belt_anchorage"],

    "P_failure_given_belt_use":
        ["belt_use", "belt_failure"],

    "P_availability_given_use_anchorage":
        ["belt_use", "belt_anchorage", "belt_availability"],

    "P_seat_performance_given_position_track_type":
        ["seat_position", "seat_track_position",
         "seat_type", "seat_performance"]
}

created = []

for name, cols in models.items():

    target = cols[-1]
    parents = cols[:-1]

    if len(parents) == 0:

        table = (
            df[target]
            .value_counts(normalize=True)
            .rename("probability")
            .reset_index()
        )

        table.columns = [target, "probability"]

    else:

        counts = (
            df.groupby(cols, dropna=False)
              .size()
              .reset_index(name="count")
        )

        totals = (
            counts.groupby(parents)["count"]
            .transform("sum")
        )

        counts["probability"] = (
            counts["count"] / totals
        )

        table = counts

    filename = name + ".csv"

    table.to_csv(
        out / filename,
        index=False
    )

    created.append(filename)

# ---------------------------------------------------------
# Model definition
# ---------------------------------------------------------

definition = pd.DataFrame([
    ["seat_position", "ROOT_STATE"],
    ["occupant_role", "CONDITIONAL_ON_SEAT_POSITION"],
    ["seat_track_position", "CONDITIONAL_ON_SEAT_POSITION_AND_ROLE"],
    ["seat_type", "CONDITIONAL_ON_SEAT_POSITION_AND_TRACK"],
    ["posture", "CONDITIONAL_ON_POSITION_TRACK_TYPE"],
    ["belt_use", "CONDITIONAL_ON_POSITION_ROLE_TRACK_TYPE"],
    ["belt_anchorage", "CONDITIONAL_ON_SEAT_TYPE_AND_BELT_USE"],
    ["belt_failure", "CONDITIONAL_ON_BELT_USE"],
    ["belt_availability", "CONDITIONAL_ON_BELT_USE_AND_ANCHORAGE"],
    ["seat_performance", "CONDITIONAL_ON_POSITION_TRACK_TYPE"]
], columns=["feature", "dependency_structure"])

definition.to_csv(
    out / "occupant_state_dependency_model.csv",
    index=False
)

print("\n==============================================")
print("DEPENDENCY-AWARE OCCUPANT STATE MODEL")
print("==============================================")

print("OBSERVATIONS:", len(df))
print("MODEL FACTORS:", len(models))

print("\nMODEL STRUCTURE:")
print(definition.to_string(index=False))

print("\nFILES CREATED:")
for f in created:
    print(out / f)

print(out / "occupant_state_dependency_model.csv")
