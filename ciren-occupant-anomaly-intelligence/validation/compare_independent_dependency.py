import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(
    base / "ciren_empirical_occupant_state.csv"
)

features = [
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

for c in features:
    df[c] = df[c].fillna(
        "NOT_COLLECTED/UNKNOWN"
    ).astype(str)


# ============================================================
# 1. INDEPENDENT MODEL
# ============================================================

independent_probability = np.ones(len(df))

for c in features:

    probabilities = (
        df[c]
        .value_counts(normalize=True)
    )

    independent_probability *= (
        df[c].map(probabilities)
    )


df["independent_probability"] = independent_probability

df["independent_nll"] = -np.log(
    df["independent_probability"]
)


# ============================================================
# 2. DEPENDENCY-AWARE MODEL
# ============================================================

models = {

    "seat_position": (
        "P_seat_position.csv",
        [],
        "seat_position"
    ),

    "occupant_role": (
        "P_role_given_seat_position.csv",
        ["seat_position"],
        "occupant_role"
    ),

    "seat_track_position": (
        "P_track_given_seat_role.csv",
        ["seat_position", "occupant_role"],
        "seat_track_position"
    ),

    "seat_type": (
        "P_type_given_seat_track.csv",
        ["seat_position", "seat_track_position"],
        "seat_type"
    ),

    "posture": (
        "P_posture_given_position_track_type.csv",
        [
            "seat_position",
            "seat_track_position",
            "seat_type"
        ],
        "posture"
    ),

    "belt_use": (
        "P_belt_use_given_position_role_track_type.csv",
        [
            "seat_position",
            "occupant_role",
            "seat_track_position",
            "seat_type"
        ],
        "belt_use"
    ),

    "belt_anchorage": (
        "P_anchorage_given_type_use.csv",
        ["seat_type", "belt_use"],
        "belt_anchorage"
    ),

    "belt_failure": (
        "P_failure_given_belt_use.csv",
        ["belt_use"],
        "belt_failure"
    ),

    "belt_availability": (
        "P_availability_given_use_anchorage.csv",
        ["belt_use", "belt_anchorage"],
        "belt_availability"
    ),

    "seat_performance": (
        "P_seat_performance_given_position_track_type.csv",
        [
            "seat_position",
            "seat_track_position",
            "seat_type"
        ],
        "seat_performance"
    )
}


def load_lookup(filename, parents, target):

    table = pd.read_csv(base / filename)

    lookup = {}

    for _, row in table.iterrows():

        key = tuple(
            str(row[p])
            for p in parents
        )

        key = key + (str(row[target]),)

        lookup[key] = float(
            row["probability"]
        )

    return lookup


lookups = {
    name: load_lookup(
        filename,
        parents,
        target
    )
    for name, (
        filename,
        parents,
        target
    ) in models.items()
}


# ============================================================
# 3. CALCULATE DEPENDENCY MODEL PROBABILITY
# ============================================================

dependency_probability = []

for _, row in df.iterrows():

    p = 1.0

    p *= lookups["seat_position"].get(
        (row["seat_position"],),
        0
    )

    p *= lookups["occupant_role"].get(
        (
            row["seat_position"],
            row["occupant_role"]
        ),
        0
    )

    p *= lookups["seat_track_position"].get(
        (
            row["seat_position"],
            row["occupant_role"],
            row["seat_track_position"]
        ),
        0
    )

    p *= lookups["seat_type"].get(
        (
            row["seat_position"],
            row["seat_track_position"],
            row["seat_type"]
        ),
        0
    )

    p *= lookups["posture"].get(
        (
            row["seat_position"],
            row["seat_track_position"],
            row["seat_type"],
            row["posture"]
        ),
        0
    )

    p *= lookups["belt_use"].get(
        (
            row["seat_position"],
            row["occupant_role"],
            row["seat_track_position"],
            row["seat_type"],
            row["belt_use"]
        ),
        0
    )

    p *= lookups["belt_anchorage"].get(
        (
            row["seat_type"],
            row["belt_use"],
            row["belt_anchorage"]
        ),
        0
    )

    p *= lookups["belt_failure"].get(
        (
            row["belt_use"],
            row["belt_failure"]
        ),
        0
    )

    p *= lookups["belt_availability"].get(
        (
            row["belt_use"],
            row["belt_anchorage"],
            row["belt_availability"]
        ),
        0
    )

    p *= lookups["seat_performance"].get(
        (
            row["seat_position"],
            row["seat_track_position"],
            row["seat_type"],
            row["seat_performance"]
        ),
        0
    )

    dependency_probability.append(p)


df["dependency_probability"] = dependency_probability

df["dependency_nll"] = -np.log(
    df["dependency_probability"]
)


# ============================================================
# 4. INFORMATION GAIN
# ============================================================

df["nll_improvement"] = (
    df["independent_nll"]
    - df["dependency_nll"]
)

mean_independent = df["independent_nll"].mean()
mean_dependency = df["dependency_nll"].mean()

mean_improvement = df["nll_improvement"].mean()

relative_improvement = (
    mean_improvement /
    mean_independent *
    100
)


# ============================================================
# 5. REPORT
# ============================================================

summary = pd.DataFrame([
    {
        "metric": "observations",
        "value": len(df)
    },
    {
        "metric": "independent_mean_NLL",
        "value": mean_independent
    },
    {
        "metric": "dependency_mean_NLL",
        "value": mean_dependency
    },
    {
        "metric": "mean_NLL_improvement",
        "value": mean_improvement
    },
    {
        "metric": "relative_NLL_improvement_percent",
        "value": relative_improvement
    }
])

summary.to_csv(
    base / "independent_vs_dependency_summary.csv",
    index=False
)

df.to_csv(
    base / "independent_vs_dependency_validation.csv",
    index=False
)


print("\n==============================================")
print("INDEPENDENT vs DEPENDENCY-AWARE MODEL")
print("==============================================")

print(
    summary.to_string(index=False)
)

print("\nINTERPRETATION:")

if mean_improvement > 0:
    print(
        "DEPENDENCY MODEL PROVIDES BETTER "
        "LIKELIHOOD THAN INDEPENDENT MODEL."
    )
else:
    print(
        "DEPENDENCY MODEL DOES NOT IMPROVE "
        "LIKELIHOOD OVER INDEPENDENT MODEL."
    )

print("\nFILES CREATED:")
print(
    base / "independent_vs_dependency_summary.csv"
)
print(
    base / "independent_vs_dependency_validation.csv"
)

