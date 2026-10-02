import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
data = pd.read_csv(
    "results/plant1_merged_clean.csv"
)

# Convert timestamp back to datetime
data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

# Plot AC power
plt.figure(figsize=(12, 5))

plt.plot(
    data["DATE_TIME"],
    data["AC_POWER"]
)

plt.xlabel("Time")
plt.ylabel("AC Power")
plt.title("Plant 1 Solar Power Generation")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()