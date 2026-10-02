import pandas as pd

# Load our cleaned dataset
data = pd.read_csv(
    "results/plant1_merged_clean.csv"
)

print("========== CLEAN DATASET ==========")

print("Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nData types:")
print(data.dtypes)

print("\nBasic statistics:")
print(data.describe())