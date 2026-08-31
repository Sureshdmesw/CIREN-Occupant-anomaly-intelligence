import pandas as pd

src = "oa.sas7bdat"
out_file = r"D:\Datasets\occupant_position_ground_truth\ciren_occupant_normalized.csv"

df = pd.read_sas(src, format="sas7bdat", encoding="latin1")

seatpos = {
    11:"FRONT LEFT SIDE",
    12:"FRONT MIDDLE",
    13:"FRONT RIGHT SIDE",
    14:"FRONT OTHER",
    15:"FRONT ON/IN LAP",
    19:"UNKNOWN SEAT, FRONT ROW",
    21:"SECOND LEFT",
    22:"SECOND MIDDLE",
    23:"SECOND RIGHT",
    24:"SECOND OTHER",
    25:"SECOND ON/IN LAP",
    29:"UNKNOWN SEAT, SECOND ROW",
    31:"THIRD LEFT",
    32:"THIRD MIDDLE",
    33:"THIRD RIGHT",
    34:"THIRD OTHER",
    35:"THIRD ON/IN LAP",
    39:"UNKNOWN SEAT, THIRD ROW"
}

role = {
    1:"DRIVER",
    2:"PASSENGER"
}

seattrack = {
    0:"OCCUPANT NOT SEATED OR NO SEAT",
    1:"NOT ADJUSTABLE SEAT TRACK",
    2:"SEAT AT FORWARD MOST POSITION",
    3:"SEAT BETWEEN FORWARD AND MIDDLE",
    4:"SEAT AT MIDDLE TRACK POSITION",
    5:"SEAT BETWEEN MIDDLE AND REAR",
    6:"SEAT AT REAR MOST POSITION"
}

posture = {
    0:"NORMAL POSTURE",
    1:"KNEELING ON SEAT",
    2:"LYING ON SEAT",
    3:"KNEELING IN FRONT OF SEAT",
    4:"SITTING SIDEWAYS",
    5:"SITTING ON CONSOLE",
    6:"LYING - SEAT BACK",
    7:"BRACING WITH FEET",
    8:"OTHER ABNORMAL POSTURE"
}

seattype = {
    0:"OCCUPANT NOT SEATED OR NO SEAT",
    1:"BUCKET",
    2:"BUCKET WITH FOLD BACK",
    3:"BENCH",
    4:"BENCH WITH SEPARATE BACK CUSHIONS",
    5:"BENCH WITH FOLD BACK",
    6:"SPLIT BENCH WITH SEPARATE BACK CUSHIONS",
    7:"SPLIT BENCH WITH FOLD BACK",
    8:"PEDESTAL",
    9:"BOX MOUNTED"
}

manavail = {
    0:"NONE AVAILABLE",
    1:"BELT REMOVED/DESTROYED",
    2:"SHOULDER BELT",
    3:"LAP BELT",
    4:"LAP AND SHOULDER BELT",
    5:"BELT AVAILABLE - TYPE UNKNOWN",
    6:"SHOULDER BELT - LAP BELT DESTROYED",
    7:"LAP BELT - SHOULDER BELT DESTROYED",
    8:"OTHER BELT"
}

manuse = {
    0:"NONE USED/AVAILABLE",
    1:"INOPERATIVE",
    2:"SHOULDER BELT",
    3:"LAP BELT",
    4:"LAP AND SHOULDER",
    5:"TYPE UNKNOWN",
    8:"OTHER BELT",
    12:"SHOULDER WITH CHILD SEAT",
    13:"LAP WITH CHILD SEAT",
    14:"LAP AND SHOULDER WITH CHILD SEAT",
    15:"OTHER CHILD SEAT BELT"
}

manfail = {
    0:"NOT USED/AVAILABLE",
    1:"NO FAILURE",
    2:"TORN WEBBING",
    3:"BROKEN BUCKLE/LATCH",
    4:"UPPER ANCHORAGE SEPARATION",
    5:"OTHER ANCHOR SEPARATION",
    6:"BROKEN RETRACTOR",
    7:"COMBINATION",
    8:"OTHER FAILURE"
}

beltanch = {
    0:"NO SHOULDER BELT",
    1:"NO UPPER ANCHOR",
    2:"FULL UP",
    3:"MID POSITION",
    4:"FULL DOWN",
    5:"POSITION UNKNOWN"
}

seatperf = {
    0:"NOT SEATED/NO SEAT",
    1:"NO FAILURE",
    2:"ADJUSTERS FAILED",
    3:"FOLD LOCK FAIL",
    4:"TRACK/ANCHOR FAILURE",
    5:"DEFORMED BY OCCUPANT",
    6:"DEFORMED BY INTRUSION",
    7:"COMBINATION",
    10:"DEFORMED BY CARGO",
    11:"DEFORMED BY OTHER OCCUPANT"
}

out = pd.DataFrame()

out["ciren_id"] = df["CIRENID"]
out["occupant_id"] = df["OCCUPANTID"]
out["vehicle_id"] = df["VEHICLEID"]
out["seat_location_id"] = df["SEATLOCID"]

out["seat_position_code"] = df["SEATPOS"]
out["seat_position"] = df["SEATPOS"].map(seatpos).fillna("UNKNOWN/UNMAPPED")

out["occupant_role_code"] = df["ROLE"]
out["occupant_role"] = df["ROLE"].map(role).fillna("UNKNOWN")

out["seat_track_code"] = df["SEATRACK"]
out["seat_track_position"] = df["SEATRACK"].map(seattrack).fillna("NOT COLLECTED/UNKNOWN")

out["posture_code"] = df["POSTURE"]
out["posture"] = df["POSTURE"].map(posture).fillna("NOT COLLECTED/UNKNOWN")

out["seat_type_code"] = df["SEATTYPE"]
out["seat_type"] = df["SEATTYPE"].map(seattype).fillna("UNKNOWN/UNMAPPED")

out["seat_performance_code"] = df["SEATPERF"]
out["seat_performance"] = df["SEATPERF"].map(seatperf).fillna("UNKNOWN/UNMAPPED")

out["belt_availability_code"] = df["MANAVAIL"]
out["belt_availability"] = df["MANAVAIL"].map(manavail).fillna("NOT COLLECTED/UNKNOWN")

out["belt_use_code"] = df["MANUSE"]
out["belt_use"] = df["MANUSE"].map(manuse).fillna("NOT COLLECTED/UNKNOWN")

out["belt_failure_code"] = df["MANFAIL"]
out["belt_failure"] = df["MANFAIL"].map(manfail).fillna("NOT COLLECTED/UNKNOWN")

out["belt_anchorage_code"] = df["BELTANCH"]
out["belt_anchorage"] = df["BELTANCH"].map(beltanch).fillna("NOT COLLECTED/UNKNOWN")

out["age"] = df["AGE"]
out["height"] = df["HEIGHT"]
out["weight"] = df["WEIGHT"]
out["sex_code"] = df["SEX"]

# Explicit provenance
out["source_type"] = "CIREN_OBSERVED"
out["source_reference"] = "NHTSA CIREN Public Data v1.2.1 / OA"
out["occupant_presence_basis"] = "CIREN_OA_RECORD"
out["geometry_status"] = "NOT_PROVIDED_BY_OA"

out.to_csv(out_file, index=False)

print("CREATED:", out_file)
print("ROWS:", len(out))
print("COLUMNS:", len(out.columns))
print("\nSeat positions:")
print(out["seat_position"].value_counts(dropna=False).to_string())
print("\nOccupant roles:")
print(out["occupant_role"].value_counts(dropna=False).to_string())
print("\nPostures:")
print(out["posture"].value_counts(dropna=False).to_string())
