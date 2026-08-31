import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

base = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase2_analysis"
)

src = base / "ciren_empirical_occupant_state.csv"

summary_file = (
    base / "backoff_smoothing_comparison.csv"
)

records_file = (
    base / "backoff_smoothing_validation_records.csv"
)


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
# REPRODUCIBLE SPLIT
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
# SMOOTHED PROBABILITY TABLE
# ============================================================

def build_table(data, parents, target, alpha):

    target_states = sorted(
        data[target].unique()
    )

    K = len(target_states)

    tables = {}

    # --------------------------------------------------------
    # Root probability
    # --------------------------------------------------------

    if len(parents) == 0:

        counts = (
            data[target]
            .value_counts()
            .reindex(
                target_states,
                fill_value=0
            )
        )

        total = counts.sum()

        probs = (
            counts + alpha
        ) / (
            total + alpha * K
        )

        for state in target_states:

            tables[
                (state,)
            ] = float(
                probs[state]
            )

        return tables, target_states


    # --------------------------------------------------------
    # Conditional probability
    # --------------------------------------------------------

    counts = (
        data
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
        .sum()
        .reset_index(name="total")
    )

    count_lookup = {}

    for _, row in counts.iterrows():

        key = tuple(
            str(row[c])
            for c in parents + [target]
        )

        count_lookup[key] = float(
            row["count"]
        )

    for _, row in totals.iterrows():

        parent_key = tuple(
            str(row[c])
            for c in parents
        )

        total = float(row["total"])

        denominator = (
            total + alpha * K
        )

        for state in target_states:

            key = parent_key + (state,)

            count = count_lookup.get(
                key,
                0.0
            )

            tables[key] = (
                count + alpha
            ) / denominator

    return tables, target_states


# ============================================================
# BUILD BACKOFF MODELS
# ============================================================

def build_backoff_model(
    data,
    parents,
    target,
    alpha
):

    # Full context
    full_table, states = build_table(
        data,
        parents,
        target,
        alpha
    )

    # Progressive backoff tables
    backoff_tables = []

    for level in range(
        len(parents),
        -1,
        -1
    ):

        current_parents = parents[:level]

        table, _ = build_table(
            data,
            current_parents,
            target,
            alpha
        )

        backoff_tables.append(
            (
                current_parents,
                table
            )
        )

    return states, backoff_tables


# ============================================================
# LOOKUP WITH BACKOFF
# ============================================================

def probability_with_backoff(
    row,
    parents,
    target,
    states,
    backoff_tables
):

    target_value = str(
        row[target]
    )

    # Most specific context first
    for current_parents, table in backoff_tables:

        key = tuple(
            str(row[c])
            for c in current_parents
        ) + (target_value,)

        if key in table:

            return table[key], len(
                current_parents
            )

    # --------------------------------------------------------
    # Absolute fallback
    # --------------------------------------------------------

    return 1.0 / len(states), 0


# ============================================================
# INDEPENDENT BASELINE
# ============================================================

ind_probability = np.ones(
    len(test)
)

for c in features:

    p = (
        train[c]
        .value_counts(
            normalize=True
        )
    )

    ind_probability *= (
        test[c]
        .map(p)
        .fillna(0)
        .to_numpy()
    )

ind_nll = np.where(
    ind_probability > 0,
    -np.log(ind_probability),
    np.inf
)

finite_ind = np.isfinite(
    ind_nll
)

ind_mean_nll = (
    ind_nll[finite_ind].mean()
)


# ============================================================
# TEST BACKOFF MODELS
# ============================================================

alphas = [
    0.001,
    0.01,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0
]

results = []

best_records = None
best_nll = np.inf
best_alpha = None


for alpha in alphas:

    print(
        "\n------------------------------------------"
    )

    print(
        "ALPHA =",
        alpha
    )

    print(
        "------------------------------------------"
    )

    models = {}

    for name, (
        parents,
        target
    ) in structures.items():

        states, backoff_tables = (
            build_backoff_model(
                train,
                parents,
                target,
                alpha
            )
        )

        models[name] = (
            parents,
            target,
            states,
            backoff_tables
        )


    probabilities = []

    backoff_levels = []

    for _, row in test.iterrows():

        p = 1.0

        levels = []

        for name in structures:

            parents, target, states, tables = (
                models[name]
            )

            prob, level = (
                probability_with_backoff(
                    row,
                    parents,
                    target,
                    states,
                    tables
                )
            )

            p *= prob

            levels.append(level)

        probabilities.append(p)

        backoff_levels.append(
            sum(levels)
        )


    probabilities = np.array(
        probabilities
    )

    nll = -np.log(
        probabilities
    )

    mean_nll = nll.mean()

    zero_cases = (
        probabilities <= 0
    ).sum()

    improvement = (
        ind_mean_nll -
        mean_nll
    )

    relative = (
        improvement /
        ind_mean_nll *
        100
    )

    results.append({

        "alpha": alpha,

        "training_observations":
            len(train),

        "test_observations":
            len(test),

        "dependency_mean_NLL":
            mean_nll,

        "zero_probability_cases":
            zero_cases,

        "coverage_percent":
            (
                probabilities > 0
            ).mean() * 100,

        "NLL_improvement":
            improvement,

        "relative_NLL_improvement_percent":
            relative,

        "mean_total_backoff_level":
            np.mean(backoff_levels)

    })


    print(
        "Mean NLL:",
        mean_nll
    )

    print(
        "Zero probability:",
        zero_cases
    )

    print(
        "Coverage:",
        round(
            (
                probabilities > 0
            ).mean() * 100,
            3
        ),
        "%"
    )

    print(
        "Relative improvement:",
        round(
            relative,
            3
        ),
        "%"
    )


    if mean_nll < best_nll:

        best_nll = mean_nll

        best_alpha = alpha

        best_records = test.copy()

        best_records[
            "backoff_probability"
        ] = probabilities

        best_records[
            "backoff_nll"
        ] = nll

        best_records[
            "backoff_alpha"
        ] = alpha


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "dependency_mean_NLL"
)

results_df.to_csv(
    summary_file,
    index=False
)

best_records.to_csv(
    records_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

best = results_df.iloc[0]

print(
    "\n=============================================="
)

print(
    "BACKOFF-SMOOTHED OCCUPANT STATE VALIDATION"
)

print(
    "=============================================="
)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\n=============================================="
)

print(
    "BEST MODEL"
)

print(
    "=============================================="
)

print(
    "Best alpha:",
    best_alpha
)

print(
    "Best NLL:",
    best_nll
)

print(
    "NLL improvement:",
    best["NLL_improvement"]
)

print(
    "Relative improvement:",
    best[
        "relative_NLL_improvement_percent"
    ],
    "%"
)

print(
    "Zero probability:",
    best[
        "zero_probability_cases"
    ]
)

print(
    "Coverage:",
    best[
        "coverage_percent"
    ],
    "%"
)

print(
    "\nFILES CREATED:"
)

print(summary_file)
print(records_file)

