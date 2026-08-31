import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# PHASE 9C — SYNTHETIC OCCUPANT EVENT GENERATION
# ============================================================

base = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase2_analysis"
)

out = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase9_event_generation"
)

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
# BUILD CONDITIONAL DISTRIBUTIONS DIRECTLY FROM CIREN
# ============================================================

conditional_tables = {}

for name, (parents, target) in structures.items():

    cols = parents + [target]

    counts = (
        df.groupby(
            cols,
            dropna=False
        )
        .size()
        .reset_index(name="count")
    )

    if parents:

        totals = (
            counts
            .groupby(parents)["count"]
            .transform("sum")
        )

    else:

        totals = counts["count"].sum()

    counts["probability"] = (
        counts["count"] / totals
    )

    conditional_tables[name] = counts

# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(2026)

N = 10000

events = []

start_time = datetime(
    2026,
    1,
    1,
    0,
    0,
    0
)

# ============================================================
# SAMPLING FUNCTION
# ============================================================

def sample_state(
    table,
    parents,
    target,
    current
):

    if parents:

        mask = np.ones(
            len(table),
            dtype=bool
        )

        for p in parents:

            mask &= (
                table[p].astype(str)
                == str(current[p])
            )

        subset = table[mask]

        # Back off progressively if the exact
        # conditioning context does not exist.

        if len(subset) == 0:

            for k in range(
                len(parents) - 1,
                -1,
                -1
            ):

                if k == 0:

                    subset = table

                else:

                    reduced = parents[:k]

                    mask = np.ones(
                        len(table),
                        dtype=bool
                    )

                    for p in reduced:

                        mask &= (
                            table[p].astype(str)
                            == str(current[p])
                        )

                    subset = table[mask]

                if len(subset) > 0:
                    break

    else:

        subset = table

    values = subset[target].astype(str).values
    probabilities = subset["probability"].values

    probabilities = (
        probabilities /
        probabilities.sum()
    )

    return rng.choice(
        values,
        p=probabilities
    )

# ============================================================
# GENERATE EVENTS
# ============================================================

for i in range(N):

    event = {}

    event["event_time"] = (
        start_time +
        timedelta(
            seconds=i
        )
    ).isoformat()

    event["event_id"] = (
        "OCC-"
        + str(i + 1).zfill(8)
    )

    event["vehicle_id"] = (
        "SIM-"
        + str(
            rng.integers(
                100000,
                999999
            )
        )
    )

    event["occupant_id"] = (
        "OCC-"
        + str(
            rng.integers(
                10000000,
                99999999
            )
        )
    )

    event["seat_location_id"] = (
        "SEAT-"
        + str(
            rng.integers(
                1000,
                9999
            )
        )
    )

    # --------------------------------------------------------
    # Sample dependency chain
    # --------------------------------------------------------

    for name, (
        parents,
        target
    ) in structures.items():

        table = conditional_tables[name]

        event[target] = sample_state(
            table,
            parents,
            target,
            event
        )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    event["source_type"] = (
        "SYNTHETIC_FROM_CIREN_EMPIRICAL_MODEL"
    )

    event["source_reference"] = (
        "NHTSA CIREN Public Data v1.2.1 / OA"
    )

    event["model_type"] = (
        "DEPENDENCY_AWARE_EMPIRICAL"
    )

    event["model_version"] = (
        "CIREN-DAM-v1"
    )

    event["geometry_status"] = (
        "NOT_PROVIDED"
    )

    event["temporal_status"] = (
        "SYNTHETIC_EVENT_TIME"
    )

    events.append(event)

# ============================================================
# SAVE
# ============================================================

events_df = pd.DataFrame(events)

event_file = (
    out /
    "synthetic_occupant_events_10000.csv"
)

events_df.to_csv(
    event_file,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([

    [
        "events_generated",
        len(events_df)
    ],

    [
        "unique_vehicles",
        events_df["vehicle_id"].nunique()
    ],

    [
        "unique_occupants",
        events_df["occupant_id"].nunique()
    ],

    [
        "source_type",
        events_df["source_type"].iloc[0]
    ],

    [
        "model_type",
        events_df["model_type"].iloc[0]
    ],

    [
        "model_version",
        events_df["model_version"].iloc[0]
    ],

    [
        "temporal_status",
        events_df["temporal_status"].iloc[0]
    ],

    [
        "geometry_status",
        events_df["geometry_status"].iloc[0]
    ]

],
columns=[
    "metric",
    "value"
])

summary.to_csv(
    out /
    "phase9C_generation_summary.csv",
    index=False
)

print("\n==============================================")
print("PHASE 9C — SYNTHETIC EVENT GENERATION")
print("==============================================")

print(
    "EVENTS GENERATED:",
    len(events_df)
)

print(
    "UNIQUE VEHICLES:",
    events_df["vehicle_id"].nunique()
)

print(
    "UNIQUE OCCUPANTS:",
    events_df["occupant_id"].nunique()
)

print("\nFIRST 10 EVENTS:")

print(
    events_df[
        [
            "event_time",
            "event_id",
            "vehicle_id",
            "occupant_id",
            "seat_position",
            "occupant_role",
            "seat_track_position",
            "posture",
            "seat_type",
            "belt_use",
            "belt_failure"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nFILES CREATED:")

print(event_file)

print(
    out /
    "phase9C_generation_summary.csv"
)

