import pandas as pd

print("Loading original dataset...")

# Load original CSV file
df = pd.read_csv("data/US_Accidents.csv")

print("Dataset loaded")
print("Initial shape:", df.shape)

# Drop columns with too many missing values
df = df.drop(columns=["End_Lat", "End_Lng"])

# Convert time columns
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")

# Remove rows with missing time
df = df.dropna(subset=["Start_Time", "End_Time"])

# Fill numeric missing values
numeric_cols = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"
]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values
categorical_cols = [
    "City", "Timezone", "Wind_Direction",
    "Weather_Condition", "Sunrise_Sunset",
    "Civil_Twilight", "Nautical_Twilight",
    "Astronomical_Twilight"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature engineering
df["Hour"] = df["Start_Time"].dt.hour
df["Weekday"] = df["Start_Time"].dt.day_name()
df["Month"] = df["Start_Time"].dt.month_name()

# Save cleaned data
df.to_parquet("US_Accidents_clean.parquet", engine="pyarrow")

print("Cleaned data saved successfully!")
print("Final shape:", df.shape)
