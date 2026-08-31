import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

base = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

src = base / "ciren_empirical_occupant_state.csv"

out_summary = base / "smoothed_model_comparison.csv"
out_records = base / "smoothed_model_validation_records.csv"


# ============================================================
# LOAD DATA
# ============================================================

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
# INDEPENDENT BASELINE
# ============================================================

independent_probability = np.ones(len(test))

for c in features:

    probabilities = (
        train[c]
        .value_counts(normalize=True)
    )

    independent_probability *= (
        test[c]
        .map(probabilities)
        .fillna(0)
        .to_numpy()
    )

test["independent_probability"] = (
    independent_probability
)

test["independent_nll"] = np.where(
    independent_probability > 0,
    -np.log(independent_probability),
    np.inf
)


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
        [
            "seat_position",
            "occupant_role"
        ],
        "seat_track_position"
    ),

    "seat_type": (
        [
            "seat_position",
            "seat_track_position"
        ],
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
        [
            "seat_type",
            "belt_use"
        ],
        "belt_anchorage"
    ),

    "belt_failure": (
        [
            "belt_use"
        ],
        "belt_failure"
    ),

    "belt_availability": (
        [
            "belt_use",
            "belt_anchorage"
        ],
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


# ============================================================
# DIRICHLET-SMOOTHED CONDITIONAL MODEL
# ============================================================

def build_smoothed_model(train, parents, target, alpha):

    # All possible target states in the training data
    target_states = sorted(
        train[target]
        .astype(str)
        .unique()
    )

    K = len(target_states)

    if len(parents) == 0:

        counts = (
            train[target]
            .value_counts()
            .reindex(
                target_states,
                fill_value=0
            )
        )

        total = counts.sum()

        probabilities = (
            (counts + alpha)
            /
            (total + alpha * K)
        )

        lookup = {}

        for state in target_states:

            lookup[(state,)] = float(
                probabilities[state]
            )

        return lookup, target_states

    # Observed parent/target combinations
    counts = (
        train
        .groupby(
            parents + [target],
            dropna=False
        )
        .size()
        .reset_index(name="count")
    )

    # Parent-context totals
    totals = (
        counts
        .groupby(parents)["count"]
        .sum()
        .reset_index(name="total")
    )

    # Create lookup for observed combinations
    count_lookup = {}

    for _, row in counts.iterrows():

        key = tuple(
            str(row[c])
            for c in parents + [target]
        )

        count_lookup[key] = float(
            row["count"]
        )

    total_lookup = {}

    for _, row in totals.iterrows():

        key = tuple(
            str(row[c])
            for c in parents
        )

        total_lookup[key] = float(
            row["total"]
        )

    lookup = {}

    # IMPORTANT:
    # Generate every target state for every
    # observed parent context.
    for parent_key, total in total_lookup.items():

        denominator = (
            total +
            alpha * K
        )

        for target_state in target_states:

            key = (
                parent_key +
                (target_state,)
            )

            count = count_lookup.get(
                key,
                0.0
            )

            probability = (
                count + alpha
            ) / denominator

            lookup[key] = probability

    return lookup, target_states


# ============================================================
# TEST MULTIPLE SMOOTHING PARAMETERS
# ============================================================

alphas = [
    0.001,
    0.01,
    0.1,
    0.5,
    1.0
]

results = []

record_outputs = []


for alpha in alphas:

    print("\n------------------------------------------")
    print("ALPHA =", alpha)
    print("------------------------------------------")

    models = {}

    for name, (parents, target) in structures.items():

        models[name], _ = build_smoothed_model(
            train,
            parents,
            target,
            alpha
        )


    # ========================================================
    # CALCULATE TEST PROBABILITY
    # ========================================================

    dependency_probability = []

    for _, row in test.iterrows():

        p = 1.0

        for name, (parents, target) in structures.items():

            key = tuple(
                str(row[c])
                for c in parents + [target]
            )

            p *= models[name].get(
                key,
                0.0
            )

        dependency_probability.append(p)


    dependency_probability = np.array(
        dependency_probability
    )

    dependency_nll = np.where(
        dependency_probability > 0,
        -np.log(dependency_probability),
        np.inf
    )


    finite = np.isfinite(
        dependency_nll
    )

    finite_nll = dependency_nll[finite]

    mean_nll = (
        finite_nll.mean()
        if len(finite_nll) > 0
        else np.inf
    )

    zero_probability = (
        dependency_probability == 0
    ).sum()

    results.append({

        "alpha": alpha,

        "training_observations":
            len(train),

        "test_observations":
            len(test),

        "dependency_mean_NLL":
            mean_nll,

        "zero_probability_cases":
            zero_probability,

        "finite_probability_cases":
            int(finite.sum()),

        "coverage_percent":
            float(
                finite.mean() * 100
            )
    })

    record_outputs.append(
        dependency_probability
    )

    print(
        "Mean dependency NLL:",
        mean_nll
    )

    print(
        "Zero-probability cases:",
        zero_probability
    )

    print(
        "Coverage:",
        round(
            finite.mean() * 100,
            3
        ),
        "%"
    )


# ============================================================
# ADD BASELINE
# ============================================================

baseline_finite = np.isfinite(
    test["independent_nll"]
)

baseline_nll = (
    test.loc[
        baseline_finite,
        "independent_nll"
    ].mean()
)

for result in results:

    result["independent_mean_NLL"] = (
        baseline_nll
    )

    result["NLL_improvement"] = (
        baseline_nll -
        result["dependency_mean_NLL"]
    )

    result["relative_NLL_improvement_percent"] = (
        result["NLL_improvement"]
        /
        baseline_nll
        *
        100
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "dependency_mean_NLL"
)

results_df.to_csv(
    out_summary,
    index=False
)


# ============================================================
# BEST ALPHA
# ============================================================

best = results_df.iloc[0]

best_alpha = best["alpha"]

best_index = alphas.index(
    best_alpha
)

test["best_smoothed_probability"] = (
    record_outputs[best_index]
)

test["best_smoothed_nll"] = np.where(
    test["best_smoothed_probability"] > 0,
    -np.log(
        test["best_smoothed_probability"]
    ),
    np.inf
)

test["best_alpha"] = best_alpha

test.to_csv(
    out_records,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==============================================")
print("DIRICHLET-SMOOTHED OCCUPANT STATE VALIDATION")
print("==============================================")

print(
    results_df.to_string(
        index=False
    )
)

print("\n==============================================")
print("BEST SMOOTHING PARAMETER")
print("==============================================")

print(
    "BEST ALPHA:",
    best_alpha
)

print(
    "BEST DEPENDENCY NLL:",
    best["dependency_mean_NLL"]
)

print(
    "NLL IMPROVEMENT:",
    best["NLL_improvement"]
)

print(
    "RELATIVE IMPROVEMENT (%):",
    best["relative_NLL_improvement_percent"]
)

print(
    "ZERO-PROBABILITY CASES:",
    best["zero_probability_cases"]
)

print(
    "COVERAGE (%):",
    best["coverage_percent"]
)

print("\nFILES CREATED:")

print(out_summary)
print(out_records)

