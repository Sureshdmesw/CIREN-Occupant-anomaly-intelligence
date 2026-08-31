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
    base / "repeated_validation_summary.csv"
)

split_file = (
    base / "repeated_validation_splits.csv"
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
        ["belt_use"],
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
# DIRICHLET-SMOOTHED TABLE
# ============================================================

def build_table(
    data,
    parents,
    target,
    alpha
):

    target_states = sorted(
        data[target].unique()
    )

    K = len(target_states)

    tables = {}

    # Root
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

            tables[(state,)] = float(
                probs[state]
            )

        return tables, target_states


    # Conditional counts
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

    tables = {}

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
# BACKOFF MODEL
# ============================================================

def build_backoff_model(
    data,
    parents,
    target,
    alpha
):

    states = sorted(
        data[target].unique()
    )

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
# BACKOFF LOOKUP
# ============================================================

def probability_with_backoff(
    row,
    parents,
    target,
    states,
    tables
):

    target_value = str(
        row[target]
    )

    for current_parents, table in tables:

        key = tuple(
            str(row[c])
            for c in current_parents
        ) + (target_value,)

        if key in table:

            return (
                table[key],
                len(current_parents)
            )

    return (
        1.0 / len(states),
        0
    )


# ============================================================
# VALIDATION PARAMETERS
# ============================================================

alpha = 0.1

seeds = [
    11,
    22,
    33,
    44,
    55,
    66,
    77,
    88,
    99,
    111
]

results = []


# ============================================================
# REPEATED VALIDATION
# ============================================================

for seed in seeds:

    print(
        "\n=========================================="
    )

    print(
        "VALIDATION SEED:",
        seed
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    indices = np.arange(
        len(df)
    )

    rng.shuffle(indices)

    split = int(
        len(df) * 0.80
    )

    train_idx = indices[:split]
    test_idx = indices[split:]

    train = df.iloc[
        train_idx
    ].copy()

    test = df.iloc[
        test_idx
    ].copy()


    # --------------------------------------------------------
    # Independent baseline
    # --------------------------------------------------------

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

    finite_ind = (
        ind_probability > 0
    )

    ind_nll = np.full(
        len(test),
        np.inf
    )

    ind_nll[finite_ind] = (
        -np.log(
            ind_probability[
                finite_ind
            ]
        )
    )

    independent_mean_nll = (
        ind_nll[
            finite_ind
        ].mean()
    )


    # --------------------------------------------------------
    # Build dependency models
    # --------------------------------------------------------

    models = {}

    for name, (
        parents,
        target
    ) in structures.items():

        states, tables = (
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
            tables
        )


    # --------------------------------------------------------
    # Dependency probability
    # --------------------------------------------------------

    dependency_probabilities = []

    total_backoff_levels = []

    for _, row in test.iterrows():

        p = 1.0

        total_level = 0

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

            total_level += level

        dependency_probabilities.append(
            p
        )

        total_backoff_levels.append(
            total_level
        )


    dependency_probabilities = np.array(
        dependency_probabilities
    )

    dependency_nll = (
        -np.log(
            dependency_probabilities
        )
    )

    dependency_mean_nll = (
        dependency_nll.mean()
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    zero_cases = (
        dependency_probabilities <= 0
    ).sum()

    coverage = (
        dependency_probabilities > 0
    ).mean() * 100

    improvement = (
        independent_mean_nll -
        dependency_mean_nll
    )

    relative_improvement = (
        improvement /
        independent_mean_nll *
        100
    )

    mean_backoff = np.mean(
        total_backoff_levels
    )


    result = {

        "seed": seed,

        "training_observations":
            len(train),

        "test_observations":
            len(test),

        "independent_mean_NLL":
            independent_mean_nll,

        "dependency_mean_NLL":
            dependency_mean_nll,

        "NLL_improvement":
            improvement,

        "relative_NLL_improvement_percent":
            relative_improvement,

        "zero_probability_cases":
            zero_cases,

        "coverage_percent":
            coverage,

        "mean_total_backoff_level":
            mean_backoff
    }

    results.append(
        result
    )


    print(
        "Independent NLL:",
        round(
            independent_mean_nll,
            6
        )
    )

    print(
        "Dependency NLL:",
        round(
            dependency_mean_nll,
            6
        )
    )

    print(
        "Improvement:",
        round(
            relative_improvement,
            3
        ),
        "%"
    )

    print(
        "Zero probability:",
        zero_cases
    )

    print(
        "Coverage:",
        coverage,
        "%"
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# AGGREGATE STATISTICS
# ============================================================

metrics = [
    "independent_mean_NLL",
    "dependency_mean_NLL",
    "NLL_improvement",
    "relative_NLL_improvement_percent",
    "zero_probability_cases",
    "coverage_percent",
    "mean_total_backoff_level"
]

aggregate_rows = []

for metric in metrics:

    values = results_df[
        metric
    ].astype(float)

    aggregate_rows.append({

        "metric": metric,

        "mean": values.mean(),

        "std": values.std(
            ddof=1
        ),

        "minimum": values.min(),

        "maximum": values.max(),

        "median": values.median()
    })


aggregate_df = pd.DataFrame(
    aggregate_rows
)


# ============================================================
# 95% CONFIDENCE INTERVAL FOR RELATIVE IMPROVEMENT
# ============================================================

improvements = (
    results_df[
        "relative_NLL_improvement_percent"
    ]
    .astype(float)
)

mean_improvement = (
    improvements.mean()
)

std_improvement = (
    improvements.std(
        ddof=1
    )
)

n = len(improvements)

standard_error = (
    std_improvement /
    np.sqrt(n)
)

ci_low = (
    mean_improvement -
    1.96 *
    standard_error
)

ci_high = (
    mean_improvement +
    1.96 *
    standard_error
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    split_file,
    index=False
)

aggregate_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n=============================================="
)

print(
    "10× REPEATED HELD-OUT VALIDATION"
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
    "AGGREGATE RESULTS"
)

print(
    "=============================================="
)

print(
    aggregate_df.to_string(
        index=False
    )
)

print(
    "\n=============================================="
)

print(
    "RELATIVE NLL IMPROVEMENT"
)

print(
    "=============================================="
)

print(
    "Mean:",
    mean_improvement,
    "%"
)

print(
    "Std:",
    std_improvement,
    "%"
)

print(
    "95% CI:",
    ci_low,
    "%",
    "to",
    ci_high,
    "%"
)

print(
    "\n=============================================="
)

print(
    "ROBUSTNESS CHECK"
)

print(
    "=============================================="
)

if (
    (results_df["zero_probability_cases"] == 0)
    .all()
    and
    (
        results_df[
            "relative_NLL_improvement_percent"
        ] > 0
    ).all()
):

    print(
        "PASS:"
    )

    print(
        "Dependency model improves over "
        "independent baseline on ALL splits."
    )

else:

    print(
        "REVIEW REQUIRED:"
    )

    print(
        "At least one split does not satisfy "
        "the robustness criteria."
    )


print(
    "\nFILES CREATED:"
)

print(summary_file)
print(split_file)

