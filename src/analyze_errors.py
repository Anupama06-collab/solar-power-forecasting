import pandas as pd

print("========== FORECAST ERROR ANALYSIS ==========")

# Load forecasts
data = pd.read_csv(
    "results/solar_forecasts.csv"
)

data["DATE_TIME"] = pd.to_datetime(
    data["DATE_TIME"]
)

# Sort by largest error
worst_errors = data.sort_values(
    "ABS_ERROR",
    ascending=False
)

print("\n========== TOP 20 LARGEST ERRORS ==========")

print(
    worst_errors[
        [
            "DATE_TIME",
            "ACTUAL_AC_POWER",
            "PREDICTED_AC_POWER",
            "ABS_ERROR"
        ]
    ].head(20).to_string(index=False)
)

# Basic error statistics
print("\n========== ERROR STATISTICS ==========")

print(
    worst_errors["ABS_ERROR"].describe()
)

# Count large errors
print("\n========== LARGE ERROR COUNTS ==========")

print(
    "Errors > 1000:",
    (data["ABS_ERROR"] > 1000).sum()
)

print(
    "Errors > 5000:",
    (data["ABS_ERROR"] > 5000).sum()
)

print(
    "Errors > 10000:",
    (data["ABS_ERROR"] > 10000).sum()
)

print("\nError analysis completed successfully!")