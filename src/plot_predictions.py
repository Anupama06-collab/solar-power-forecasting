import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

print("========== GENERATING PREDICTION PLOT ==========")

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

target = "TARGET_AC_POWER"

# Time-series split
split_index = int(len(data) * 0.80)

train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Plot
plt.figure(figsize=(14, 6))

plt.plot(
    test_data["DATE_TIME"],
    y_test,
    label="Actual AC Power"
)

plt.plot(
    test_data["DATE_TIME"],
    predictions,
    label="Predicted AC Power"
)

plt.xlabel("Date and Time")
plt.ylabel("AC Power")
plt.title("Actual vs Predicted Solar Power")

plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()

# Save plot
output_path = "results/actual_vs_predicted.png"

plt.savefig(output_path, dpi=300)

print("Plot saved to:")
print(output_path)

plt.show()