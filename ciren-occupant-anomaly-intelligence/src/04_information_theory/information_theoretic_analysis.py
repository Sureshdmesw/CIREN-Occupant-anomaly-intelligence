import pandas as pd
import numpy as np
from pathlib import Path

src = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis\ciren_empirical_occupant_state.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

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
    df[c] = df[c].fillna("NOT_COLLECTED/UNKNOWN").astype(str)

# ============================================================
# 1. SHANNON ENTROPY
# ============================================================

entropy_rows = []

for c in features:

    p = df[c].value_counts(normalize=True)

    entropy = -(p * np.log2(p)).sum()

    entropy_rows.append({
        "feature": c,
        "entropy_bits": entropy,
        "unique_states": df[c].nunique(),
        "observations": len(df)
    })

entropy_df = pd.DataFrame(entropy_rows)

entropy_df = entropy_df.sort_values(
    "entropy_bits",
    ascending=False
)

entropy_df.to_csv(
    out / "occupant_state_entropy.csv",
    index=False
)

# ============================================================
# 2. MUTUAL INFORMATION
# ============================================================

mi_rows = []

for i, a in enumerate(features):

    for b in features[i+1:]:

        joint = pd.crosstab(
            df[a],
            df[b],
            normalize=True
        )

        pa = joint.sum(axis=1)
        pb = joint.sum(axis=0)

        mi = 0.0

        for x in joint.index:

            for y in joint.columns:

                pxy = joint.loc[x, y]

                if pxy > 0:

                    mi += pxy * np.log2(
                        pxy / (pa[x] * pb[y])
                    )

        mi_rows.append({
            "feature_a": a,
            "feature_b": b,
            "mutual_information_bits": mi
        })

mi_df = pd.DataFrame(mi_rows)

# ============================================================
# 3. NORMALIZED MUTUAL INFORMATION
# ============================================================

normalized_values = []

for _, row in mi_df.iterrows():

    a = row["feature_a"]
    b = row["feature_b"]

    ha = entropy_df.loc[
        entropy_df["feature"] == a,
        "entropy_bits"
    ].iloc[0]

    hb = entropy_df.loc[
        entropy_df["feature"] == b,
        "entropy_bits"
    ].iloc[0]

    mi = row["mutual_information_bits"]

    if ha > 0 and hb > 0:
        nmi = mi / np.sqrt(ha * hb)
    else:
        nmi = 0.0

    normalized_values.append(nmi)

mi_df["normalized_mi"] = normalized_values

mi_df = mi_df.sort_values(
    "mutual_information_bits",
    ascending=False
)

mi_df.to_csv(
    out / "occupant_state_mutual_information.csv",
    index=False
)

# ============================================================
# 4. DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("OCCUPANT STATE INFORMATION-THEORETIC ANALYSIS")
print("==============================================")

print("\nSHANNON ENTROPY:")
print(
    entropy_df.to_string(index=False)
)

print("\nTOP 20 MUTUAL-INFORMATION PAIRS:")
print(
    mi_df.head(20).to_string(index=False)
)

print("\nFILES CREATED:")
print(out / "occupant_state_entropy.csv")
print(out / "occupant_state_mutual_information.csv")

print("\nTOTAL FEATURE PAIRS:", len(mi_df))
