import pandas as pd
import joblib

print("========== SOLAR POWER FORECAST ==========")

# Load saved model
model = joblib.load(
    "models/random_forest_solar.pkl"
)

# Load feature list
features = joblib.load(
    "models/model_features.pkl"
)

print("Saved model loaded successfully!")

# Load dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

# Use the first sample from the test period
split_index = int(len(data) * 0.80)

test_data = data.iloc[split_index:].copy()

sample = test_data.iloc[[0]]

# Prepare input
X = sample[features]

# Forecast
forecast = model.predict(X)[0]

# Actual value for validation
actual = sample["TARGET_AC_POWER"].iloc[0]

timestamp = sample["DATE_TIME"].iloc[0]

print("\n========== FORECAST ==========")

print("Forecast time:", timestamp)
print(f"Predicted AC Power: {forecast:.2f}")

print("\n========== VALIDATION ==========")

print(f"Actual AC Power:    {actual:.2f}")
print(f"Prediction Error:   {abs(actual - forecast):.2f}")

print("\nForecast completed successfully!")