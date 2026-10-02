import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "results/solar_improved_dataset.csv"


print("========== XGBOOST TRAINING ==========")


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

print("Total rows:", len(df))


# ============================================================
# 2. SORT BY TIME
# ============================================================

df = df.sort_values("DATE_TIME").reset_index(drop=True)


# ============================================================
# 3. TIME-BASED TRAIN/TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\n========== TIME SERIES SPLIT ==========")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining period:")
print("Start:", train_df["DATE_TIME"].min())
print("End  :", train_df["DATE_TIME"].max())

print("\nTesting period:")
print("Start:", test_df["DATE_TIME"].min())
print("End  :", test_df["DATE_TIME"].max())


# ============================================================
# 4. FEATURES
# ============================================================

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

    # Improved features
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


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


print("\n========== ML DATA ==========")

print("Number of features:", len(features))
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 5. CREATE XGBOOST MODEL
# ============================================================

print("\n========== CREATING XGBOOST MODEL ==========")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 6. TRAIN
# ============================================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 7. PREDICTION
# ============================================================

print("\n========== PREDICTION ==========")

predictions = model.predict(X_test)

print("Number of predictions:", len(predictions))


print("\nFirst 10 predictions:")

for actual, predicted in zip(
    y_test.iloc[:10],
    predictions[:10]
):
    print(
        f"Actual: {actual:.2f} | "
        f"Predicted: {predicted:.2f}"
    )


# ============================================================
# 8. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n========== XGBOOST RESULTS ==========")

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# 9. BASELINE COMPARISON
# ============================================================

print("\n========== BASELINE COMPARISON ==========")

print("Random Forest baseline:")
print("MAE : 1000.94")
print("RMSE: 2029.44")
print("R²  : 0.9352")

print("\nXGBoost:")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# 10. IMPROVEMENT
# ============================================================

baseline_mae = 1000.94
baseline_rmse = 2029.44
baseline_r2 = 0.9352


mae_improvement = (
    (baseline_mae - mae)
    / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - rmse)
    / baseline_rmse
) * 100

r2_improvement = (
    (r2 - baseline_r2)
    / baseline_r2
) * 100


print("\n========== IMPROVEMENT ==========")

print(
    f"MAE improvement : {mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: {rmse_improvement:.2f}%"
)

print(
    f"R² improvement  : {r2_improvement:.2f}%"
)


print("\n========== COMPLETED ==========")