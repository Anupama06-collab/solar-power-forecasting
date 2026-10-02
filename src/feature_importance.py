import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

print("========== FEATURE IMPORTANCE ==========")

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

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE RESULTS ==========")

print(importance_df.to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

output_path = "results/feature_importance.png"

plt.savefig(output_path, dpi=300)

print("\nFeature importance plot saved to:")
print(output_path)

plt.show()