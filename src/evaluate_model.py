import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

print("========== MODEL EVALUATION ==========")

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

# Metrics
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("Number of test samples:", len(y_test))

print("\n========== RESULTS ==========")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")