import pandas as pd
import matplotlib.pyplot as plt

print("========== PLOTTING FORECASTS ==========")

# Load forecast results
data = pd.read_csv(
    "results/solar_forecasts.csv"
)

data["DATE_TIME"] = pd.to_datetime(
    data["DATE_TIME"]
)

print("Forecast records:", len(data))

# Create plot
plt.figure(figsize=(14, 6))

plt.plot(
    data["DATE_TIME"],
    data["ACTUAL_AC_POWER"],
    label="Actual AC Power"
)

plt.plot(
    data["DATE_TIME"],
    data["PREDICTED_AC_POWER"],
    label="Predicted AC Power"
)

plt.xlabel("Date and Time")
plt.ylabel("AC Power")
plt.title("Actual vs Predicted Solar Power")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

# Save figure
output_path = "results/forecast_plot.png"

plt.savefig(output_path, dpi=300)

plt.close()

print("\nPlot saved successfully!")
print("Path:", output_path)