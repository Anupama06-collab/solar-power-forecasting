import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

print("========== SAVING FINAL SOLAR MODEL ==========")

# Load ML dataset
df = pd.read_csv("results/solar_ml_dataset.csv")

# Convert timestamp
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

# Features used by the validated baseline model
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
split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]

X_train = train[features]
y_train = train[target]

print("Training samples:", len(X_train))
print("Number of features:", len(features))

# Create final Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining final Random Forest...")
model.fit(X_train, y_train)

print("Training completed!")

# Save final model
joblib.dump(
    model,
    "models/final_solar_model.pkl"
)

# Save feature list
joblib.dump(
    features,
    "models/final_model_features.pkl"
)

print("\n========== FINAL MODEL SAVED ==========")

print("Model:")
print("models/final_solar_model.pkl")

print("\nFeature list:")
print("models/final_model_features.pkl")

print("\nFinal features:")
for feature in features:
    print("-", feature)

print("\nSUCCESS: Final solar forecasting model is ready!")