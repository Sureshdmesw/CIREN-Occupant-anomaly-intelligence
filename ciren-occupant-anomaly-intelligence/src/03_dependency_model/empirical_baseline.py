import pandas as pd
from pathlib import Path

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

rows = []

for c in cols:
    counts = df[c].value_counts(dropna=False)

    for value, count in counts.items():
        rows.append({
            "variable": c,
            "state": value,
            "count": int(count),
            "probability": round(count / len(df), 6),
            "percentage": round(count / len(df) * 100, 4)
        })

baseline = pd.DataFrame(rows)

baseline.to_csv(
    out / "occupant_state_empirical_baseline.csv",
    index=False
)

print("CREATED:", out / "occupant_state_empirical_baseline.csv")
print("BASELINE ROWS:", len(baseline))
