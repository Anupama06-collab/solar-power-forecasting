import pandas as pd
import joblib
import os

print("========== GENERATING TEST PERIOD FORECASTS ==========")

# Load model
model = joblib.load(
    "models/random_forest_solar.pkl"
)

# Load feature list
features = joblib.load(
    "models/model_features.pkl"
)

print("Model loaded successfully!")

# Load ML dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

# Same 80/20 time-series split
split_index = int(len(data) * 0.80)

test_data = data.iloc[split_index:].copy()

print("Test samples:", len(test_data))

# Prepare features
X_test = test_data[features]

# Generate predictions
predictions = model.predict(X_test)

# Create forecast results
forecast_results = pd.DataFrame({
    "DATE_TIME": test_data["DATE_TIME"],
    "ACTUAL_AC_POWER": test_data["TARGET_AC_POWER"],
    "PREDICTED_AC_POWER": predictions
})

# Calculate absolute error
forecast_results["ABS_ERROR"] = (
    forecast_results["ACTUAL_AC_POWER"]
    - forecast_results["PREDICTED_AC_POWER"]
).abs()

# Create results directory if needed
os.makedirs("results", exist_ok=True)

# Save results
output_path = "results/solar_forecasts.csv"

forecast_results.to_csv(
    output_path,
    index=False
)

print("\n========== FORECAST SUMMARY ==========")

print("Number of forecasts:", len(forecast_results))

print("\nFirst 10 forecasts:")
print(
    forecast_results.head(10).to_string(index=False)
)

print("\nAverage absolute error:",
      forecast_results["ABS_ERROR"].mean())

print("\nMaximum absolute error:",
      forecast_results["ABS_ERROR"].max())

print("\nForecast file saved successfully!")
print("Path:", output_path)