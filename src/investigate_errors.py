import pandas as pd

print("========== INVESTIGATING WORST PREDICTIONS ==========")

# Load ML dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

data["DATE_TIME"] = pd.to_datetime(
    data["DATE_TIME"]
)

# Same time-series split used for testing
split_index = int(len(data) * 0.80)

test_data = data.iloc[split_index:].copy()

# Load forecast results
forecasts = pd.read_csv(
    "results/solar_forecasts.csv"
)

forecasts["DATE_TIME"] = pd.to_datetime(
    forecasts["DATE_TIME"]
)

# Merge forecast errors with original features
merged = test_data.merge(
    forecasts[
        [
            "DATE_TIME",
            "PREDICTED_AC_POWER",
            "ABS_ERROR"
        ]
    ],
    on="DATE_TIME",
    how="inner"
)

# Sort by largest error
worst = merged.sort_values(
    "ABS_ERROR",
    ascending=False
)

# Features we want to inspect
columns_to_show = [
    "DATE_TIME",
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HOUR",
    "MINUTE",
    "AC_POWER_LAG_1",
    "AC_POWER_LAG_2",
    "AC_POWER_LAG_3",
    "AC_POWER",
    "TARGET_AC_POWER",
    "PREDICTED_AC_POWER",
    "ABS_ERROR"
]

print("\n========== TOP 20 WORST PREDICTIONS WITH FEATURES ==========")

print(
    worst[columns_to_show]
    .head(20)
    .to_string(index=False)
)

print("\n========== INVESTIGATION COMPLETED ==========")