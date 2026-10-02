import pandas as pd
import joblib

print("========== TESTING FINAL SOLAR MODEL ==========")

# Load final model
model = joblib.load("models/final_solar_model.pkl")

# Load feature list
features = joblib.load("models/final_model_features.pkl")

print("Final model loaded successfully!")

print("\nExpected features:")
for feature in features:
    print("-", feature)

# Load dataset
df = pd.read_csv("results/solar_ml_dataset.csv")
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

# Use first test sample
test_sample = df.iloc[2522]

# Prepare input
X_test = pd.DataFrame(
    [[test_sample[feature] for feature in features]],
    columns=features
)

# Prediction
prediction = model.predict(X_test)[0]

actual = test_sample["TARGET_AC_POWER"]

print("\n========== PREDICTION ==========")
print(f"Forecast Time       : {test_sample['DATE_TIME']}")
print(f"Actual AC Power     : {actual:.2f}")
print(f"Predicted AC Power  : {prediction:.2f}")
print(f"Absolute Error      : {abs(actual - prediction):.2f}")

print("\n========== FINAL MODEL STATUS ==========")
print("SUCCESS: Final model prediction completed!")
print("Model is ready for Kafka integration.")