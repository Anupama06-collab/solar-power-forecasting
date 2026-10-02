import pandas as pd
import joblib

print("========== LOADING SAVED MODEL ==========")

# Load model
model = joblib.load(
    "models/random_forest_solar.pkl"
)

# Load feature list
features = joblib.load(
    "models/model_features.pkl"
)

print("Model loaded successfully!")

print("\nExpected features:")
for feature in features:
    print("-", feature)

# Load test dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

# Use the first test sample
split_index = int(len(data) * 0.80)

test_data = data.iloc[split_index:].copy()

sample = test_data.iloc[[0]]

X_sample = sample[features]

# Make prediction
prediction = model.predict(X_sample)

actual = sample["TARGET_AC_POWER"].iloc[0]

print("\n========== SAVED MODEL PREDICTION ==========")

print(f"Actual AC Power: {actual:.2f}")
print(f"Predicted AC Power: {prediction[0]:.2f}")

print("\nSaved model test completed successfully!")