import pandas as pd
import numpy as np
from pathlib import Path

base = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase6_anomaly"
)

out = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase7_validation"
)

src = base / "empirical_occupant_state_anomaly_scores.csv"

df = pd.read_csv(src)

# ============================================================
# 1. ANOMALY SCORE DISTRIBUTION
# ============================================================

score = df["anomaly_score_bits"]

quantiles = [
    0.50,
    0.75,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
    1.00
]

rows = []

for q in quantiles:

    rows.append({
        "quantile": q,
        "percentile": q * 100,
        "anomaly_score_bits": score.quantile(q)
    })

thresholds = pd.DataFrame(rows)

thresholds.to_csv(
    out / "anomaly_score_quantiles.csv",
    index=False
)

# ============================================================
# 2. THRESHOLD SENSITIVITY
# ============================================================

threshold_percentiles = [
    90,
    95,
    97.5,
    99,
    99.5,
    99.9
]

threshold_rows = []

for p in threshold_percentiles:

    threshold = np.percentile(
        score,
        p
    )

    records = (
        score >= threshold
    ).sum()

    threshold_rows.append({
        "threshold_percentile": p,
        "threshold_score_bits": threshold,
        "records_above_threshold": records,
        "percentage_of_dataset": (
            records / len(df) * 100
        )
    })

threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    out / "anomaly_threshold_sensitivity.csv",
    index=False
)

# ============================================================
# 3. FEATURE CONTRIBUTION ANALYSIS
# ============================================================

surprisal_cols = [
    c for c in df.columns
    if c.endswith("_surprisal_bits")
]

contribution_rows = []

for c in surprisal_cols:

    feature = c.replace(
        "_surprisal_bits",
        ""
    )

    values = df[c]

    contribution_rows.append({
        "feature": feature,
        "mean_surprisal_bits": values.mean(),
        "median_surprisal_bits": values.median(),
        "std_surprisal_bits": values.std(),
        "max_surprisal_bits": values.max(),
        "records_nonzero": (
            values > 0
        ).sum(),
        "nonzero_rate_percent": (
            (values > 0).mean() * 100
        )
    })

contribution_df = pd.DataFrame(
    contribution_rows
)

contribution_df = contribution_df.sort_values(
    "mean_surprisal_bits",
    ascending=False
)

contribution_df.to_csv(
    out / "global_feature_surprisal_profile.csv",
    index=False
)

# ============================================================
# 4. EXTREME-ANOMALY CONTRIBUTION PROFILE
# ============================================================

extreme = df[
    df["anomaly_tier"] == "EXTREME"
].copy()

extreme_rows = []

for c in surprisal_cols:

    feature = c.replace(
        "_surprisal_bits",
        ""
    )

    values = extreme[c]

    extreme_rows.append({
        "feature": feature,
        "mean_surprisal_bits": values.mean(),
        "median_surprisal_bits": values.median(),
        "max_surprisal_bits": values.max(),
        "total_surprisal_bits": values.sum()
    })

extreme_df = pd.DataFrame(
    extreme_rows
)

extreme_df["total_contribution_percent"] = (
    extreme_df["total_surprisal_bits"]
    /
    extreme_df["total_surprisal_bits"].sum()
    * 100
)

extreme_df = extreme_df.sort_values(
    "total_contribution_percent",
    ascending=False
)

extreme_df.to_csv(
    out / "extreme_anomaly_contribution_profile.csv",
    index=False
)

# ============================================================
# 5. ANOMALY TIER × DOMINANT CONTRIBUTOR
# ============================================================

tier_contributor = pd.crosstab(
    df["anomaly_tier"],
    df["dominant_contributor"]
)

tier_contributor.to_csv(
    out / "anomaly_tier_by_dominant_contributor.csv"
)

# ============================================================
# 6. TOP ANOMALY STATE PATTERNS
# ============================================================

state_cols = [
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

top_extreme = (
    extreme
    .groupby(state_cols)
    .agg(
        observations=("anomaly_score_bits", "size"),
        mean_anomaly_score_bits=(
            "anomaly_score_bits",
            "mean"
        ),
        maximum_anomaly_score_bits=(
            "anomaly_score_bits",
            "max"
        )
    )
    .reset_index()
    .sort_values(
        "mean_anomaly_score_bits",
        ascending=False
    )
)

top_extreme.to_csv(
    out / "extreme_state_patterns.csv",
    index=False
)

# ============================================================
# 7. GLOBAL SUMMARY
# ============================================================

summary = pd.DataFrame([
    [
        "total_observations",
        len(df)
    ],
    [
        "mean_anomaly_score_bits",
        score.mean()
    ],
    [
        "median_anomaly_score_bits",
        score.median()
    ],
    [
        "std_anomaly_score_bits",
        score.std()
    ],
    [
        "p90_anomaly_score_bits",
        score.quantile(0.90)
    ],
    [
        "p95_anomaly_score_bits",
        score.quantile(0.95)
    ],
    [
        "p99_anomaly_score_bits",
        score.quantile(0.99)
    ],
    [
        "p999_anomaly_score_bits",
        score.quantile(0.999)
    ],
    [
        "extreme_records",
        len(extreme)
    ],
    [
        "extreme_percentage",
        len(extreme) / len(df) * 100
    ]
], columns=["metric", "value"])

summary.to_csv(
    out / "phase7_validation_summary.csv",
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("PHASE 7 — ANOMALY MODEL CHARACTERIZATION")
print("==============================================")

print("\nGLOBAL SUMMARY:")
print(
    summary.to_string(index=False)
)

print("\nTHRESHOLD SENSITIVITY:")
print(
    threshold_df.to_string(index=False)
)

print("\nGLOBAL FEATURE SURPRISAL:")
print(
    contribution_df.to_string(index=False)
)

print("\nEXTREME-ANOMALY CONTRIBUTION:")
print(
    extreme_df.to_string(index=False)
)

print("\nANOMALY TIER × DOMINANT CONTRIBUTOR:")
print(
    tier_contributor.to_string()
)

print("\nTOP EXTREME STATE PATTERNS:")
print(
    top_extreme.head(20).to_string(index=False)
)

print("\nFILES CREATED:")

for f in [
    "anomaly_score_quantiles.csv",
    "anomaly_threshold_sensitivity.csv",
    "global_feature_surprisal_profile.csv",
    "extreme_anomaly_contribution_profile.csv",
    "anomaly_tier_by_dominant_contributor.csv",
    "extreme_state_patterns.csv",
    "phase7_validation_summary.csv"
]:
    print(out / f)

