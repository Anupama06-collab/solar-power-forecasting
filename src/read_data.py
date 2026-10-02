import pandas as pd

# Load Plant 1 data
generation = pd.read_csv(
    "data/Plant_1_Generation_Data.csv"
)

weather = pd.read_csv(
    "data/Plant_1_Weather_Sensor_Data.csv"
)

print("========== GENERATION DATA ==========")
print("Columns:")
print(generation.columns.tolist())

print("\nFirst 5 rows:")
print(generation.head())


print("\n\n========== WEATHER DATA ==========")
print("Columns:")
print(weather.columns.tolist())

print("\nFirst 5 rows:")
print(weather.head())

print("\n========== GENERATION STRUCTURE ==========")

print("Unique timestamps:",
      generation["DATE_TIME"].nunique())

print("Unique inverters:",
      generation["SOURCE_KEY"].nunique())

print("\n========== WEATHER STRUCTURE ==========")

print("Unique timestamps:",
      weather["DATE_TIME"].nunique())

print("Unique weather sensors:",
      weather["SOURCE_KEY"].nunique())

print("\n========== CONVERTING TIMESTAMPS ==========")

generation["DATE_TIME"] = pd.to_datetime(
    generation["DATE_TIME"],
    dayfirst=True
)

weather["DATE_TIME"] = pd.to_datetime(
    weather["DATE_TIME"]
)

print("Generation timestamp type:")
print(generation["DATE_TIME"].dtype)

print("Weather timestamp type:")
print(weather["DATE_TIME"].dtype)

print("\nGeneration first timestamp:")
print(generation["DATE_TIME"].iloc[0])

print("Weather first timestamp:")
print(weather["DATE_TIME"].iloc[0])

print("\n========== AGGREGATING INVERTERS ==========")

generation_plant = (
    generation
    .groupby("DATE_TIME")
    .agg(
        AC_POWER=("AC_POWER", "sum"),
        DC_POWER=("DC_POWER", "sum")
    )
    .reset_index()
)

print("Original generation rows:", len(generation))
print("Plant-level rows:", len(generation_plant))

print("\nFirst 5 plant-level records:")
print(generation_plant.head())

print("\n========== PLANT POWER CHECK ==========")

print("Maximum AC power:",
      generation_plant["AC_POWER"].max())

print("Minimum AC power:",
      generation_plant["AC_POWER"].min())

print("Average AC power:",
      generation_plant["AC_POWER"].mean())

print("\n========== PREPARING WEATHER DATA ==========")

weather_plant = weather[
    [
        "DATE_TIME",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]
].copy()

print("Weather rows:", len(weather_plant))
print("Weather columns:")
print(weather_plant.columns.tolist())

print("\nFirst 5 weather records:")
print(weather_plant.head())

print("\n========== MERGING GENERATION + WEATHER ==========")

merged_data = pd.merge(
    generation_plant,
    weather_plant,
    on="DATE_TIME",
    how="inner"
)

print("Generation timestamps:", len(generation_plant))
print("Weather timestamps:", len(weather_plant))
print("Merged rows:", len(merged_data))

print("\nMerged columns:")
print(merged_data.columns.tolist())

print("\nFirst 5 merged records:")
print(merged_data.head())

print("\n========== MISSING VALUE CHECK ==========")

print(merged_data.isnull().sum())

print("\n========== TIME ORDER CHECK ==========")

print("First timestamp:")
print(merged_data["DATE_TIME"].min())

print("Last timestamp:")
print(merged_data["DATE_TIME"].max())

print("Data is sorted:",
      merged_data["DATE_TIME"].is_monotonic_increasing)

print("\n========== TIME INTERVAL CHECK ==========")

time_difference = (
    merged_data["DATE_TIME"]
    .diff()
    .dropna()
)

print("Most common time interval:")
print(time_difference.value_counts().head())

print("\n========== TIME GAPS ==========")

time_difference = merged_data["DATE_TIME"].diff()

gaps = merged_data.loc[
    time_difference > pd.Timedelta(minutes=15),
    ["DATE_TIME"]
].copy()

gaps["GAP_FROM_PREVIOUS"] = time_difference[
    time_difference > pd.Timedelta(minutes=15)
].values

print("Number of gaps:", len(gaps))
print(gaps)

print("\n========== SAVING CLEAN DATA ==========")

output_path = "results/plant1_merged_clean.csv"

merged_data.to_csv(
    output_path,
    index=False
)

print("Saved successfully to:")
print(output_path)

print("Final rows:", len(merged_data))
print("Final columns:", len(merged_data.columns))