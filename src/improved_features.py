import pandas as pd
import numpy as np

INPUT_FILE = "results/solar_ml_dataset.csv"
OUTPUT_FILE = "results/solar_improved_dataset.csv"


print("========== IMPROVED FEATURE ENGINEERING ==========")

# Load existing ML dataset
df = pd.read_csv(INPUT_FILE)

# Convert timestamp
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

print("Original shape:", df.shape)


# ============================================================
# 1. CYCLICAL TIME FEATURES
# ============================================================

print("\n========== CREATING CYCLICAL TIME FEATURES ==========")

# Convert time into continuous hour
time_in_hours = df["HOUR"] + df["MINUTE"] / 60

# Daily cycle
df["HOUR_SIN"] = np.sin(2 * np.pi * time_in_hours / 24)
df["HOUR_COS"] = np.cos(2 * np.pi * time_in_hours / 24)

# Yearly/seasonal cycle
df["DAY_SIN"] = np.sin(2 * np.pi * df["DAY_OF_YEAR"] / 365)
df["DAY_COS"] = np.cos(2 * np.pi * df["DAY_OF_YEAR"] / 365)


# ============================================================
# 2. TEMPERATURE DIFFERENCE
# ============================================================

print("\n========== CREATING TEMPERATURE FEATURES ==========")

df["TEMP_DIFFERENCE"] = (
    df["MODULE_TEMPERATURE"] -
    df["AMBIENT_TEMPERATURE"]
)


# ============================================================
# 3. IRRADIATION CHANGE
# ============================================================

print("\n========== CREATING IRRADIATION FEATURES ==========")

df["IRRADIATION_LAG_1"] = df["IRRADIATION"].shift(1)

df["IRRADIATION_CHANGE"] = (
    df["IRRADIATION"] -
    df["IRRADIATION_LAG_1"]
)


# ============================================================
# 4. POWER CHANGE FEATURES
# ============================================================

print("\n========== CREATING POWER CHANGE FEATURES ==========")

df["POWER_CHANGE_1"] = (
    df["AC_POWER"] -
    df["AC_POWER_LAG_1"]
)

df["POWER_CHANGE_2"] = (
    df["AC_POWER_LAG_1"] -
    df["AC_POWER_LAG_2"]
)


# ============================================================
# 5. ROLLING POWER FEATURES
# ============================================================

print("\n========== CREATING ROLLING POWER FEATURES ==========")

df["POWER_ROLLING_MEAN_3"] = (
    df["AC_POWER"]
    .rolling(window=3)
    .mean()
)

df["POWER_ROLLING_STD_3"] = (
    df["AC_POWER"]
    .rolling(window=3)
    .std()
)


# ============================================================
# 6. REMOVE ROWS WITH NEW MISSING VALUES
# ============================================================

print("\n========== HANDLING MISSING VALUES ==========")

before = len(df)

df = df.dropna().reset_index(drop=True)

after = len(df)

print("Rows before:", before)
print("Rows after :", after)
print("Rows removed:", before - after)


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

print("\n========== IMPROVED DATASET ==========")

print("Shape:", df.shape)

print("\nNew columns:")

new_columns = [
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_SIN",
    "DAY_COS",
    "TEMP_DIFFERENCE",
    "IRRADIATION_LAG_1",
    "IRRADIATION_CHANGE",
    "POWER_CHANGE_1",
    "POWER_CHANGE_2",
    "POWER_ROLLING_MEAN_3",
    "POWER_ROLLING_STD_3"
]

for column in new_columns:
    print("-", column)


print("\nFirst 5 rows:")
print(
    df[
        [
            "DATE_TIME",
            "AC_POWER",
            "IRRADIATION",
            "HOUR_SIN",
            "HOUR_COS",
            "TEMP_DIFFERENCE",
            "IRRADIATION_CHANGE",
            "POWER_CHANGE_1",
            "POWER_ROLLING_MEAN_3"
        ]
    ].head()
)


# ============================================================
# 8. SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== SAVING DATASET ==========")
print("Saved successfully!")
print("Path:", OUTPUT_FILE)
print("Final rows:", len(df))
print("Final columns:", len(df.columns))