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

state_frequency_file = (
    base / "occupant_state_space_statistics.csv"
)

summary_file = (
    base / "occupant_state_space_summary.csv"
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
# STATE VECTOR
# ============================================================

df["occupant_state_vector"] = (
    df[features]
    .astype(str)
    .agg("|".join, axis=1)
)


# ============================================================
# EMPIRICAL STATE FREQUENCY
# ============================================================

state_counts = (
    df["occupant_state_vector"]
    .value_counts()
    .rename("observations")
    .reset_index()
)

state_counts.columns = [
    "occupant_state_vector",
    "observations"
]

N = len(df)

state_counts["empirical_probability"] = (
    state_counts["observations"] / N
)


# ============================================================
# SELF-INFORMATION
# ============================================================

state_counts["self_information_bits"] = (
    -np.log2(
        state_counts["empirical_probability"]
    )
)


# ============================================================
# RARITY CLASS
# ============================================================

def classify_rarity(p):

    if p >= 0.01:
        return "COMMON"

    elif p >= 0.001:
        return "UNCOMMON"

    elif p >= 0.0001:
        return "RARE"

    else:
        return "EXTREMELY_RARE"


state_counts["rarity_class"] = (
    state_counts[
        "empirical_probability"
    ]
    .apply(classify_rarity)
)


# ============================================================
# CUMULATIVE MASS
# ============================================================

state_counts = state_counts.sort_values(
    "empirical_probability",
    ascending=False
).reset_index(drop=True)

state_counts["cumulative_probability"] = (
    state_counts[
        "empirical_probability"
    ].cumsum()
)


state_counts["rank"] = (
    np.arange(
        1,
        len(state_counts) + 1
    )
)


# ============================================================
# EFFECTIVE STATE-SPACE SIZE
# ============================================================

p = state_counts[
    "empirical_probability"
].to_numpy()

shannon_entropy = (
    -np.sum(
        p * np.log2(p)
    )
)

effective_state_space = (
    2 ** shannon_entropy
)


# ============================================================
# CONCENTRATION
# ============================================================

top_10_mass = (
    state_counts
    .head(10)[
        "empirical_probability"
    ]
    .sum()
)

top_25_mass = (
    state_counts
    .head(25)[
        "empirical_probability"
    ]
    .sum()
)

top_50_mass = (
    state_counts
    .head(50)[
        "empirical_probability"
    ]
    .sum()
)

top_100_mass = (
    state_counts
    .head(100)[
        "empirical_probability"
    ]
    .sum()
)


# ============================================================
# COVERAGE OF STATE SPACE
# ============================================================

distinct_states = (
    len(state_counts)
)

singletons = (
    state_counts[
        "observations"
    ] == 1
).sum()

doubletons = (
    state_counts[
        "observations"
    ] == 2
).sum()


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([

    [
        "total_observations",
        N
    ],

    [
        "distinct_state_vectors",
        distinct_states
    ],

    [
        "state_space_entropy_bits",
        shannon_entropy
    ],

    [
        "effective_state_space_size",
        effective_state_space
    ],

    [
        "singleton_states",
        singletons
    ],

    [
        "doubleton_states",
        doubletons
    ],

    [
        "top_10_probability_mass",
        top_10_mass
    ],

    [
        "top_25_probability_mass",
        top_25_mass
    ],

    [
        "top_50_probability_mass",
        top_50_mass
    ],

    [
        "top_100_probability_mass",
        top_100_mass
    ],

    [
        "common_states",
        (
            state_counts[
                "rarity_class"
            ] == "COMMON"
        ).sum()
    ],

    [
        "uncommon_states",
        (
            state_counts[
                "rarity_class"
            ] == "UNCOMMON"
        ).sum()
    ],

    [
        "rare_states",
        (
            state_counts[
                "rarity_class"
            ] == "RARE"
        ).sum()
    ],

    [
        "extremely_rare_states",
        (
            state_counts[
                "rarity_class"
            ]
            == "EXTREMELY_RARE"
        ).sum()
    ]
],
columns=[
    "metric",
    "value"
])


# ============================================================
# SAVE
# ============================================================

state_counts.to_csv(
    state_frequency_file,
    index=False
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print(
    "\n=============================================="
)

print(
    "EMPIRICAL OCCUPANT STATE SPACE"
)

print(
    "=============================================="
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\n=============================================="
)

print(
    "TOP 25 MOST COMMON STATES"
)

print(
    "=============================================="
)

print(
    state_counts[
        [
            "rank",
            "occupant_state_vector",
            "observations",
            "empirical_probability",
            "self_information_bits",
            "rarity_class"
        ]
    ]
    .head(25)
    .to_string(index=False)
)


print(
    "\n=============================================="
)

print(
    "RAREST STATES"
)

print(
    "=============================================="
)

print(
    state_counts[
        [
            "rank",
            "occupant_state_vector",
            "observations",
            "empirical_probability",
            "self_information_bits",
            "rarity_class"
        ]
    ]
    .tail(25)
    .to_string(index=False)
)


print(
    "\nFILES CREATED:"
)

print(state_frequency_file)

print(summary_file)

