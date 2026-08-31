import pandas as pd

df = pd.read_sas(
    "oa.sas7bdat",
    format="sas7bdat",
    encoding="latin1"
)

cols = [
    "SEATPOS",
    "ROLE",
    "SEATRACK",
    "POSTURE",
    "SEATTYPE",
    "SEATPERF",
    "MANAVAIL",
    "MANUSE",
    "MANFAIL",
    "BELTANCH"
]

for c in cols:
    print("\n### " + c)
    print(df[c].value_counts(dropna=False).sort_index().to_string())
