import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency

src = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis\ciren_empirical_occupant_state.csv")
out = Path(r"D:\Datasets\occupant_position_ground_truth\phase2_analysis")

df = pd.read_csv(src)

cols = [
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

def cramers_v(x, y):
    table = pd.crosstab(x, y)

    if table.empty:
        return np.nan

    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()

    if n == 0:
        return np.nan

    phi2 = chi2 / n
    r, k = table.shape

    phi2corr = max(
        0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1)
    )

    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denominator = min(kcorr - 1, rcorr - 1)

    if denominator <= 0:
        return np.nan

    return np.sqrt(phi2corr / denominator)

results = []

for i, a in enumerate(cols):
    for b in cols[i+1:]:
        value = cramers_v(df[a], df[b])

        results.append({
            "variable_a": a,
            "variable_b": b,
            "cramers_v": round(value, 6)
        })

result = (
    pd.DataFrame(results)
    .sort_values("cramers_v", ascending=False)
)

result.to_csv(
    out / "occupant_state_dependency_matrix.csv",
    index=False
)

print("PAIRWISE RELATIONSHIPS:", len(result))
print("\nSTRONGEST RELATIONSHIPS:")
print(result.head(20).to_string(index=False))
