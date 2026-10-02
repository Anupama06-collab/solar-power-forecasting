import pandas as pd
import joblib
import matplotlib.pyplot as plt

MODEL_PATH = "models/final_solar_model.pkl"
DATA_PATH = "results/solar_improved_dataset.csv"

print("========== ACTUAL VS PREDICTED ==========")

# Load model
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

# Final model features
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

# Same 80/20 time-based split
split_index = int(len(df) * 0.80)

test_df = df.iloc[split_index:]

X_test = test_df[features]
y_test = test_df[target]

# Predictions
predictions = model.predict(X_test)

# Plot
plt.figure(figsize=(12, 6))

plt.plot(
    test_df["DATE_TIME"],
    y_test.values,
    label="Actual AC Power"
)

plt.plot(
    test_df["DATE_TIME"],
    predictions,
    label="Predicted AC Power"
)

plt.xlabel("Time")
plt.ylabel("AC Power")
plt.title("Actual vs Predicted Solar Power")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Save figure
plt.savefig("results/actual_vs_predicted.png", dpi=300)

print()
print("Graph saved successfully!")
print("Path: results/actual_vs_predicted.png")