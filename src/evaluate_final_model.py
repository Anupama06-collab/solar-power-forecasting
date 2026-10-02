import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

MODEL_PATH = "models/final_solar_model.pkl"
DATA_PATH = "results/solar_improved_dataset.csv"

print("========== FINAL MODEL EVALUATION ==========")

# Load model
model = joblib.load(MODEL_PATH)
print("Final model loaded successfully!")

# Load dataset
df = pd.read_csv(DATA_PATH)
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

# Features used by final model
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

# Time-based split
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_test = test_df[features]
y_test = test_df[target]

# Predict
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print()
print("========== FINAL MODEL RESULTS ==========")
print("Testing samples:", len(test_df))
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))

print()
print("========== EVALUATION COMPLETED ==========")