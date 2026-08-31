import pandas as pd
from pathlib import Path

base = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase2_analysis"
)

out = Path(
    r"D:\Datasets\occupant_position_ground_truth\phase9_transition_analysis"
)

src = base / "ciren_empirical_occupant_state.csv"

df = pd.read_csv(src)

print("\n==============================================")
print("PHASE 9A — TEMPORAL STRUCTURE AUDIT")
print("==============================================")

print("\nROWS:", len(df))

print("\nCOLUMNS:")
for c in df.columns:
    print(c)

print("\nPOTENTIAL TEMPORAL COLUMNS:")

temporal_keywords = [
    "time",
    "timestamp",
    "date",
    "event",
    "sequence",
    "order",
    "phase",
    "impact"
]

potential = []

for c in df.columns:

    name = c.lower()

    if any(
        k in name
        for k in temporal_keywords
    ):
        potential.append(c)

if potential:
    print(
        "\n".join(potential)
    )
else:
    print("NONE FOUND")

print("\nIDENTIFIER COLUMNS:")

identifier_keywords = [
    "vehicle",
    "occupant",
    "case",
    "ciren",
    "seat",
    "event"
]

for c in df.columns:

    name = c.lower()

    if any(
        k in name
        for k in identifier_keywords
    ):
        print(
            c,
            "unique=",
            df[c].nunique(dropna=True)
        )

print("\nROWS PER VEHICLE:")

vehicle_counts = (
    df.groupby("vehicle_id")
      .size()
)

print(
    vehicle_counts
    .describe()
    .to_string()
)

print("\nVEHICLES WITH MULTIPLE RECORDS:")

print(
    (vehicle_counts > 1).sum()
)

print("\nTOP VEHICLES BY RECORD COUNT:")

print(
    vehicle_counts
    .sort_values(
        ascending=False
    )
    .head(20)
    .to_string()
)

# ------------------------------------------------------------
# Save audit
# ------------------------------------------------------------

audit = pd.DataFrame({
    "column": df.columns,
    "dtype": [
        str(df[c].dtype)
        for c in df.columns
    ],
    "unique_values": [
        df[c].nunique(
            dropna=True
        )
        for c in df.columns
    ],
    "missing_values": [
        df[c].isna().sum()
        for c in df.columns
    ]
})

audit.to_csv(
    out /
    "phase9_temporal_structure_audit.csv",
    index=False
)

print("\nFILE CREATED:")
print(
    out /
    "phase9_temporal_structure_audit.csv"
)

