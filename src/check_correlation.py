import pandas as pd

# Load cleaned dataset
data = pd.read_csv(
    "results/plant1_merged_clean.csv"
)

print("========== CORRELATION WITH AC POWER ==========")

correlation = data[
    [
        "AC_POWER",
        "DC_POWER",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]
].corr()

print(correlation["AC_POWER"].sort_values(ascending=False))