import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

src = base / "ciren_empirical_occupant_state.csv"

df = pd.read_csv(src)

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
    df[c] = (
        df[c]
        .fillna("NOT_COLLECTED/UNKNOWN")
        .astype(str)
    )


# ============================================================
# REPRODUCIBLE 80/20 SPLIT
# ============================================================

rng = np.random.default_rng(42)

indices = np.arange(len(df))
rng.shuffle(indices)

split = int(len(df) * 0.80)

train_idx = indices[:split]
test_idx = indices[split:]

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

print("TRAIN:", len(train))
print("TEST :", len(test))


# ============================================================
# INDEPENDENT MODEL
# LEARN ONLY FROM TRAIN
# ============================================================

independent_probability = np.ones(len(test))

for c in features:

    p = train[c].value_counts(normalize=True)

    independent_probability *= (
        test[c].map(p).fillna(0)
    )

test["independent_probability"] = (
    independent_probability
)

test["independent_nll"] = np.where(
    test["independent_probability"] > 0,
    -np.log(test["independent_probability"]),
    np.inf
)


# ============================================================
# CONDITIONAL PROBABILITY TABLE BUILDER
# ============================================================

def conditional_model(train, parents, target):

    if len(parents) == 0:

        table = (
            train[target]
            .value_counts(normalize=True)
            .rename("probability")
            .reset_index()
        )

        table.columns = [target, "probability"]

        lookup = {}

        for _, row in table.iterrows():
            lookup[(str(row[target]),)] = float(
                row["probability"]
            )

        return lookup

    counts = (
        train
        .groupby(
            parents + [target],
            dropna=False
        )
        .size()
        .reset_index(name="count")
    )

    totals = (
        counts
        .groupby(parents)["count"]
        .transform("sum")
    )

    counts["probability"] = (
        counts["count"] / totals
    )

    lookup = {}

    for _, row in counts.iterrows():

        key = tuple(
            str(row[c])
            for c in parents + [target]
        )

        lookup[key] = float(
            row["probability"]
        )

    return lookup


# ============================================================
# DEPENDENCY STRUCTURE
# ============================================================

structures = {

    "seat_position": (
        [],
        "seat_position"
    ),

    "occupant_role": (
        ["seat_position"],
        "occupant_role"
    ),

    "seat_track_position": (
        ["seat_position", "occupant_role"],
        "seat_track_position"
    ),

    "seat_type": (
        ["seat_position", "seat_track_position"],
        "seat_type"
    ),

    "posture": (
        [
            "seat_position",
            "seat_track_position",
            "seat_type"
        ],
        "posture"
    ),

    "belt_use": (
        [
            "seat_position",
            "occupant_role",
            "seat_track_position",
            "seat_type"
        ],
        "belt_use"
    ),

    "belt_anchorage": (
        ["seat_type", "belt_use"],
        "belt_anchorage"
    ),

    "belt_failure": (
        ["belt_use"],
        "belt_failure"
    ),

    "belt_availability": (
        ["belt_use", "belt_anchorage"],
        "belt_availability"
    ),

    "seat_performance": (
        [
            "seat_position",
            "seat_track_position",
            "seat_type"
        ],
        "seat_performance"
    )
}


models = {}

for name, (parents, target) in structures.items():

    models[name] = conditional_model(
        train,
        parents,
        target
    )


# ============================================================
# DEPENDENCY MODEL TEST PROBABILITY
# ============================================================

dependency_probability = []

for _, row in test.iterrows():

    p = 1.0

    for name, (parents, target) in structures.items():

        key = tuple(
            str(row[c])
            for c in parents + [target]
        )

        p *= models[name].get(key, 0)

    dependency_probability.append(p)


test["dependency_probability"] = (
    dependency_probability
)

test["dependency_nll"] = np.where(
    test["dependency_probability"] > 0,
    -np.log(test["dependency_probability"]),
    np.inf
)


# ============================================================
# VALIDATION METRICS
# ============================================================

finite_ind = test[
    np.isfinite(test["independent_nll"])
]

finite_dep = test[
    np.isfinite(test["dependency_nll"])
]

mean_independent = (
    finite_ind["independent_nll"].mean()
)

mean_dependency = (
    finite_dep["dependency_nll"].mean()
)

zero_independent = (
    test["independent_probability"] == 0
).sum()

zero_dependency = (
    test["dependency_probability"] == 0
).sum()

improvement = (
    mean_independent -
    mean_dependency
)

relative_improvement = (
    improvement /
    mean_independent *
    100
)


# ============================================================
# RESULTS
# ============================================================

summary = pd.DataFrame([
    ["total_observations", len(df)],
    ["training_observations", len(train)],
    ["test_observations", len(test)],
    ["independent_test_mean_NLL", mean_independent],
    ["dependency_test_mean_NLL", mean_dependency],
    ["test_mean_NLL_improvement", improvement],
    ["test_relative_NLL_improvement_percent",
     relative_improvement],
    ["independent_zero_probability_test_cases",
     zero_independent],
    ["dependency_zero_probability_test_cases",
     zero_dependency]
], columns=["metric", "value"])


summary.to_csv(
    base / "heldout_validation_summary.csv",
    index=False
)

test.to_csv(
    base / "heldout_validation_records.csv",
    index=False
)


print("\n==============================================")
print("HELD-OUT OCCUPANT STATE VALIDATION")
print("==============================================")

print(
    summary.to_string(index=False)
)

print("\nFILES CREATED:")
print(
    base / "heldout_validation_summary.csv"
)
print(
    base / "heldout_validation_records.csv"
)

