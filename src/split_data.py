import pandas as pd

# Load ML dataset
data = pd.read_csv(
    "results/solar_ml_dataset.csv"
)

# Convert timestamp
data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

print("========== TIME SERIES SPLIT ==========")

print("Total rows:", len(data))

# 80% training, 20% testing
split_index = int(len(data) * 0.80)

train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

print("\nTraining rows:", len(train_data))
print("Testing rows:", len(test_data))

print("\nTraining period:")
print("Start:", train_data["DATE_TIME"].min())
print("End:", train_data["DATE_TIME"].max())

print("\nTesting period:")
print("Start:", test_data["DATE_TIME"].min())
print("End:", test_data["DATE_TIME"].max())

print("\n========== CHECKING TIME ORDER ==========")

if train_data["DATE_TIME"].max() < test_data["DATE_TIME"].min():
    print("SUCCESS: Training data is entirely before testing data.")
else:
    print("ERROR: Time leakage detected!")