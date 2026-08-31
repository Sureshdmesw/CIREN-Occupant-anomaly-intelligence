import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

data = pd.read_csv(
    base / "ciren_empirical_occupant_state.csv"
)

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
    data[c] = data[c].fillna("NOT_COLLECTED/UNKNOWN").astype(str)


# ============================================================
# LOAD CONDITIONAL TABLES
# ============================================================

def load_probability(filename, parents, target):

    table = pd.read_csv(base / filename)

    if parents:

        lookup = {}

        for _, row in table.iterrows():

            key = tuple(str(row[p]) for p in parents)
            key = key + (str(row[target]),)

            lookup[key] = float(row["probability"])

        return lookup

    else:

        lookup = {}

        for _, row in table.iterrows():
            lookup[str(row[target])] = float(row["probability"])

        return lookup


models = {

    "seat_position": load_probability(
        "P_seat_position.csv",
        [],
        "seat_position"
    ),

    "occupant_role": load_probability(
        "P_role_given_seat_position.csv",
        ["seat_position"],
        "occupant_role"
    ),

    "seat_track_position": load_probability(
        "P_track_given_seat_role.csv",
        ["seat_position", "occupant_role"],
        "seat_track_position"
    ),

    "seat_type": load_probability(
        "P_type_given_seat_track.csv",
        ["seat_position", "seat_track_position"],
        "seat_type"
    ),

    "posture": load_probability(
        "P_posture_given_position_track_type.csv",
        ["seat_position", "seat_track_position", "seat_type"],
        "posture"
    ),

    "belt_use": load_probability(
        "P_belt_use_given_position_role_track_type.csv",
        [
            "seat_position",
            "occupant_role",
            "seat_track_position",
            "seat_type"
        ],
        "belt_use"
    ),

    "belt_anchorage": load_probability(
        "P_anchorage_given_type_use.csv",
        ["seat_type", "belt_use"],
        "belt_anchorage"
    ),

    "belt_failure": load_probability(
        "P_failure_given_belt_use.csv",
        ["belt_use"],
        "belt_failure"
    ),

    "belt_availability": load_probability(
        "P_availability_given_use_anchorage.csv",
        ["belt_use", "belt_anchorage"],
        "belt_availability"
    ),

    "seat_performance": load_probability(
        "P_seat_performance_given_position_track_type.csv",
        ["seat_position", "seat_track_position", "seat_type"],
        "seat_performance"
    )
}


# ============================================================
# CALCULATE FACTORIZED PROBABILITY
# ============================================================

probabilities = []

for _, row in data.iterrows():

    p = 1.0

    # P(seat position)
    p *= models["seat_position"].get(
        row["seat_position"],
        0
    )

    # P(role | seat position)
    p *= models["occupant_role"].get(
        (
            row["seat_position"],
            row["occupant_role"]
        ),
        0
    )

    # P(track | seat position, role)
    p *= models["seat_track_position"].get(
        (
            row["seat_position"],
            row["occupant_role"],
            row["seat_track_position"]
        ),
        0
    )

    # P(type | seat position, track)
    p *= models["seat_type"].get(
        (
            row["seat_position"],
            row["seat_track_position"],
            row["seat_type"]
        ),
        0
    )

    # P(posture | position, track, type)
    p *= models["posture"].get(
        (
            row["seat_position"],
            row["seat_track_position"],
            row["seat_type"],
            row["posture"]
        ),
        0
    )

    # P(belt use | position, role, track, type)
    p *= models["belt_use"].get(
        (
            row["seat_position"],
            row["occupant_role"],
            row["seat_track_position"],
            row["seat_type"],
            row["belt_use"]
        ),
        0
    )

    # P(anchorage | type, belt use)
    p *= models["belt_anchorage"].get(
        (
            row["seat_type"],
            row["belt_use"],
            row["belt_anchorage"]
        ),
        0
    )

    # P(failure | belt use)
    p *= models["belt_failure"].get(
        (
            row["belt_use"],
            row["belt_failure"]
        ),
        0
    )

    # P(availability | belt use, anchorage)
    p *= models["belt_availability"].get(
        (
            row["belt_use"],
            row["belt_anchorage"],
            row["belt_availability"]
        ),
        0
    )

    # P(seat performance | position, track, type)
    p *= models["seat_performance"].get(
        (
            row["seat_position"],
            row["seat_track_position"],
            row["seat_type"],
            row["seat_performance"]
        ),
        0
    )

    probabilities.append(p)


data["factorized_probability"] = probabilities

data["negative_log_likelihood"] = np.where(
    data["factorized_probability"] > 0,
    -np.log(data["factorized_probability"]),
    np.inf
)


# ============================================================
# SUMMARY
# ============================================================

zero_probability = (
    data["factorized_probability"] == 0
).sum()

finite = data[
    np.isfinite(data["negative_log_likelihood"])
]

print("\n==============================================")
print("PROBABILISTIC MODEL VALIDATION")
print("==============================================")

print("OBSERVATIONS:", len(data))

print(
    "ZERO-PROBABILITY OBSERVATIONS:",
    zero_probability
)

print(
    "ZERO-PROBABILITY PERCENT:",
    round(zero_probability / len(data) * 100, 3),
    "%"
)

if len(finite) > 0:

    print(
        "MEAN NEGATIVE LOG-LIKELIHOOD:",
        round(finite["negative_log_likelihood"].mean(), 6)
    )

    print(
        "MEDIAN NEGATIVE LOG-LIKELIHOOD:",
        round(finite["negative_log_likelihood"].median(), 6)
    )

    print(
        "MAX FINITE NEGATIVE LOG-LIKELIHOOD:",
        round(finite["negative_log_likelihood"].max(), 6)
    )

print("\nTOP 20 HIGHEST-LIKELIHOOD STATES:")
print(
    data.sort_values(
        "factorized_probability",
        ascending=False
    )[
        [
            "seat_position",
            "occupant_role",
            "seat_track_position",
            "posture",
            "seat_type",
            "belt_use",
            "belt_failure",
            "factorized_probability",
            "negative_log_likelihood"
        ]
    ].head(20).to_string(index=False)
)

data.to_csv(
    base / "occupant_state_factorized_validation.csv",
    index=False
)

print(
    "\nCREATED:",
    base / "occupant_state_factorized_validation.csv"
)
