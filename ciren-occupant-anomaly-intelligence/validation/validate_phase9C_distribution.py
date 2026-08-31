import pandas as pd
from pathlib import Path

base = Path(
    r"D:\Datasets\occupant_position_ground_truth"
)

ciren = pd.read_csv(
    base / "phase2_analysis" /
    "ciren_empirical_occupant_state.csv"
)

synthetic = pd.read_csv(
    base / "phase9_event_generation" /
    "synthetic_occupant_events_10000.csv"
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

rows = []

for feature in features:

    c = (
        ciren[feature]
        .fillna("NOT_COLLECTED/UNKNOWN")
        .astype(str)
        .value_counts(normalize=True)
    )

    s = (
        synthetic[feature]
        .fillna("NOT_COLLECTED/UNKNOWN")
        .astype(str)
        .value_counts(normalize=True)
    )

    categories = sorted(
        set(c.index) | set(s.index)
    )

    for category in categories:

        cp = c.get(category, 0.0)
        sp = s.get(category, 0.0)

        rows.append({
            "feature": feature,
            "state": category,
            "ciren_percent": cp * 100,
            "synthetic_percent": sp * 100,
            "absolute_difference_percent":
                abs(cp - sp) * 100
        })

comparison = pd.DataFrame(rows)

comparison.to_csv(
    base / "phase9_event_generation" /
    "phase9C_ciren_vs_synthetic_distribution.csv",
    index=False
)

summary = (
    comparison
    .groupby("feature")
    .agg(
        mean_absolute_difference_percent=(
            "absolute_difference_percent",
            "mean"
        ),
        max_absolute_difference_percent=(
            "absolute_difference_percent",
            "max"
        )
    )
    .reset_index()
)

summary.to_csv(
    base / "phase9_event_generation" /
    "phase9C_distribution_validation_summary.csv",
    index=False
)

print("\n==============================================")
print("PHASE 9C — DISTRIBUTION VALIDATION")
print("==============================================")

print("\nMEAN ABSOLUTE DIFFERENCE:")
print(
    summary
    .sort_values(
        "mean_absolute_difference_percent",
        ascending=False
    )
    .to_string(index=False)
)

print("\nOVERALL MEAN ABSOLUTE DIFFERENCE:",
      comparison["absolute_difference_percent"].mean())

print("\nFILES CREATED:")

print(
    base /
    "phase9_event_generation" /
    "phase9C_ciren_vs_synthetic_distribution.csv"
)

print(
    base /
    "phase9_event_generation" /
    "phase9C_distribution_validation_summary.csv"
)
