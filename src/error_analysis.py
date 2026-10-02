import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "models/final_solar_model.pkl"
DATA_PATH = "results/solar_improved_dataset.csv"

print("========== ERROR ANALYSIS ==========")

# Load model and dataset
model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

features = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HOUR",
    "MINUTE",
    "DAY_OF_YEAR",
    "AC_POWER_LAG_1",
    "AC_POWER_LAG_2",
    "AC_POWER_LAG_3"
]

target = "AC_POWER"

# Same time-based test split
split_index = int(len(df) * 0.80)
test_df = df.iloc[split_index:].copy()

X_test = test_df[features]
y_test = test_df[target]

# Predictions
test_df["PREDICTED_AC_POWER"] = model.predict(X_test)

# Calculate errors
test_df["ABSOLUTE_ERROR"] = abs(
    test_df["AC_POWER"] - test_df["PREDICTED_AC_POWER"]
)

test_df["PERCENT_ERROR"] = np.where(
    test_df["AC_POWER"] != 0,
    (test_df["ABSOLUTE_ERROR"] / test_df["AC_POWER"]) * 100,
    0
)

print()
print("Average absolute error:",
      round(test_df["ABSOLUTE_ERROR"].mean(), 2))

print("Maximum absolute error:",
      round(test_df["ABSOLUTE_ERROR"].max(), 2))

print("Minimum absolute error:",
      round(test_df["ABSOLUTE_ERROR"].min(), 2))

# Save detailed error data
test_df[
    [
        "DATE_TIME",
        "AC_POWER",
        "PREDICTED_AC_POWER",
        "ABSOLUTE_ERROR",
        "PERCENT_ERROR"
    ]
].to_csv(
    "results/error_analysis.csv",
    index=False
)

# Plot error over time
plt.figure(figsize=(12, 6))

plt.plot(
    test_df["DATE_TIME"],
    test_df["ABSOLUTE_ERROR"],
    label="Absolute Error"
)

plt.xlabel("Time")
plt.ylabel("Absolute Error")
plt.title("Solar Power Forecast Error Analysis")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "results/error_analysis.png",
    dpi=300
)

print()
print("Error analysis saved successfully!")
print("CSV: results/error_analysis.csv")
print("Graph: results/error_analysis.png")