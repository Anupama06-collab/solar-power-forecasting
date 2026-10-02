import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "results/solar_improved_dataset.csv"


print("========== LEAKAGE-FREE FEATURE EXPERIMENT ==========")

# Load dataset
df = pd.read_csv(DATA_PATH)
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

# Only features available before prediction
features = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HOUR",
    "MINUTE",
    "DAY_OF_YEAR",
    "AC_POWER_LAG_1",
    "AC_POWER_LAG_2",
    "AC_POWER_LAG_3",
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_SIN",
    "DAY_COS",
    "TEMP_DIFFERENCE",
    "IRRADIATION_LAG_1",
    "IRRADIATION_CHANGE",
    "POWER_CHANGE_2"
]

target = "TARGET_AC_POWER"

# Time-series split
split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print("Number of features:", len(features))

print("\n========== TRAINING ==========")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")

# Prediction
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== IMPROVED MODEL RESULTS ==========")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))

print("\n========== COMPLETED ==========")