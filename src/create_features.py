import pandas as pd

# Load cleaned dataset
data = pd.read_csv(
    "results/plant1_merged_clean.csv"
)

# Convert timestamp
data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

print("========== CREATING TIME FEATURES ==========")

# Time-based features
data["HOUR"] = data["DATE_TIME"].dt.hour
data["MINUTE"] = data["DATE_TIME"].dt.minute
data["DAY_OF_YEAR"] = data["DATE_TIME"].dt.dayofyear

print("New columns:")
print(data.columns.tolist())

print("\nFirst 10 time features:")
print(
    data[
        [
            "DATE_TIME",
            "HOUR",
            "MINUTE",
            "DAY_OF_YEAR"
        ]
    ].head(10)
)

print("\n========== CREATING FORECAST TARGET ==========")

# Predict AC power 15 minutes into the future
data["TARGET_AC_POWER"] = data["AC_POWER"].shift(-1)

print("Target created.")

print("\nLast 5 records:")
print(
    data[
        [
            "DATE_TIME",
            "AC_POWER",
            "TARGET_AC_POWER"
        ]
    ].tail()
)

print("\nMissing target values:")
print(data["TARGET_AC_POWER"].isnull().sum())

print("\n========== CREATING LAG FEATURES ==========")

# Previous 15-minute AC power
data["AC_POWER_LAG_1"] = data["AC_POWER"].shift(1)

# Previous 30-minute AC power
data["AC_POWER_LAG_2"] = data["AC_POWER"].shift(2)

# Previous 45-minute AC power
data["AC_POWER_LAG_3"] = data["AC_POWER"].shift(3)

print("Lag features created.")

print("\nFirst 10 records:")
print(
    data[
        [
            "DATE_TIME",
            "AC_POWER",
            "AC_POWER_LAG_1",
            "AC_POWER_LAG_2",
            "AC_POWER_LAG_3",
            "TARGET_AC_POWER"
        ]
    ].head(10)
)

print("\n========== MISSING VALUES AFTER FEATURES ==========")

print(data.isnull().sum())

print("\n========== CREATING FINAL ML DATASET ==========")

# Remove rows where lag or target values are unavailable
ml_data = data.dropna().copy()

print("Rows before removing missing values:", len(data))
print("Rows after removing missing values:", len(ml_data))

print("\nRemaining missing values:")
print(ml_data.isnull().sum())

# Save feature-engineered dataset
output_path = "results/solar_ml_dataset.csv"

ml_data.to_csv(
    output_path,
    index=False
)

print("\nSaved ML dataset to:")
print(output_path)

print("\nFinal columns:")
print(ml_data.columns.tolist())