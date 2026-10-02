import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("========== RANDOM FOREST HYPERPARAMETER TUNING ==========")

# Load improved dataset
df = pd.read_csv("results/solar_improved_dataset.csv")
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

# Same 80/20 time-series split
split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

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
    "AC_POWER_LAG_3",
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_SIN",
    "DAY_COS",
    "TEMP_DIFFERENCE",
    "IRRADIATION_LAG_1",
    "IRRADIATION_CHANGE",
    "POWER_CHANGE_1",
    "POWER_CHANGE_2",
    "POWER_ROLLING_MEAN_3",
    "POWER_ROLLING_STD_3"
]

target = "TARGET_AC_POWER"

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# Hyperparameter combinations
configs = [
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    },
    {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    }
]

results = []

for i, config in enumerate(configs, 1):

    print("\n--------------------------------")
    print(f"MODEL {i}")
    print(config)

    model = RandomForestRegressor(
        **config,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    results.append({
        **config,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# Results
results_df = pd.DataFrame(results)

print("\n========== ALL RESULTS ==========")
print(results_df.to_string(index=False))

# Best model based on RMSE
best_index = results_df["RMSE"].idxmin()
best = results_df.loc[best_index]

print("\n========== BEST MODEL ==========")
print(best)

results_df.to_csv(
    "results/random_forest_tuning_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/random_forest_tuning_results.csv")