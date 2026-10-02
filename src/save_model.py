import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

print("========== TRAINING FINAL BASELINE MODEL ==========")

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

X_train = train_data[features]
y_train = train_data[target]

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

print("Training completed!")

# Save model
model_path = "models/random_forest_solar.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Model path:")
print(model_path)

# Save feature list
features_path = "models/model_features.pkl"

joblib.dump(features, features_path)

print("\nFeature list saved successfully!")
print("Feature list path:")
print(features_path)