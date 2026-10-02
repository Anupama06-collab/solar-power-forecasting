import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

print("========== RANDOM FOREST TRAINING ==========")

# Features
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

# X and y
X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
predictions = model.predict(X_test)

print("\n========== PREDICTION CHECK ==========")

print("Number of predictions:", len(predictions))

print("\nFirst 10 predictions:")

for i in range(10):
    print(
        f"Actual: {y_test.iloc[i]:.2f} | "
        f"Predicted: {predictions[i]:.2f}"
    )