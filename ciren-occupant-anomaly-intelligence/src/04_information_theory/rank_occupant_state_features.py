import pandas as pd
import numpy as np
from pathlib import Path

src = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis\ciren_empirical_occupant_state.csv")
mi_file = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis\occupant_state_mutual_information.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(src)
mi = pd.read_csv(mi_file)

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

# ---------------------------------------------------------
# Information connectivity of each feature
# ---------------------------------------------------------

rows = []

for feature in features:

    subset = mi[
        (mi["feature_a"] == feature) |
        (mi["feature_b"] == feature)
    ]

    raw_sum = subset["mutual_information_bits"].sum()
    normalized_sum = subset["normalized_mi"].sum()

    mean_mi = subset["mutual_information_bits"].mean()
    mean_nmi = subset["normalized_mi"].mean()

    rows.append({
        "feature": feature,
        "total_mutual_information": raw_sum,
        "total_normalized_mutual_information": normalized_sum,
        "mean_mutual_information": mean_mi,
        "mean_normalized_mutual_information": mean_nmi,
        "number_of_relationships": len(subset)
    })

ranking = pd.DataFrame(rows)

ranking = ranking.sort_values(
    "total_mutual_information",
    ascending=False
)

ranking.to_csv(
    out / "occupant_state_information_ranking.csv",
    index=False
)

# ---------------------------------------------------------
# Build preliminary model tiers
# ---------------------------------------------------------

core = [
    "seat_position",
    "seat_track_position",
    "posture",
    "seat_type"
]

restraint = [
    "belt_availability",
    "belt_use",
    "belt_failure",
    "belt_anchorage"
]

context = [
    "occupant_role",
    "seat_performance"
]

tier_rows = []

for f in core:
    tier_rows.append({
        "feature": f,
        "model_tier": "CORE_OCCUPANT_POSITION_STATE"
    })

for f in restraint:
    tier_rows.append({
        "feature": f,
        "model_tier": "RESTRAINT_STATE"
    })

for f in context:
    tier_rows.append({
        "feature": f,
        "model_tier": "CONTEXT_STRUCTURAL_STATE"
    })

tiers = pd.DataFrame(tier_rows)

final = tiers.merge(
    ranking,
    on="feature",
    how="left"
)

final = final.sort_values(
    "total_mutual_information",
    ascending=False
)

final.to_csv(
    out / "occupant_state_model_tiers.csv",
    index=False
)

print("\n==========================================")
print("INFORMATION-RANKED OCCUPANT STATE MODEL")
print("==========================================")

print("\nFEATURE INFORMATION RANKING:")
print(
    ranking.to_string(index=False)
)

print("\nMODEL TIERS:")
print(
    final[
        [
            "feature",
            "model_tier",
            "total_mutual_information",
            "mean_mutual_information"
        ]
    ].to_string(index=False)
)

print("\nFILES CREATED:")
print(out / "occupant_state_information_ranking.csv")
print(out / "occupant_state_model_tiers.csv")
