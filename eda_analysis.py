import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_palette("Set2")
plt.style.use("seaborn-v0_8-whitegrid")

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load cleaned data
df = pd.read_parquet("US_Accidents_clean.parquet")

print("Data loaded successfully")
print(df.shape)

# -----------------------------
# GRAPH 1: ACCIDENTS BY HOUR
# -----------------------------
plt.figure(figsize=(10, 5))
df['Hour'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Hour")
plt.tight_layout()
plt.savefig("outputs/accidents_by_hour.png")
plt.close()

print("Graph saved: accidents_by_hour.png")

# -----------------------------
# GRAPH 2: ACCIDENTS BY WEEKDAY
# -----------------------------
plt.figure(figsize=(10, 5))
df['Weekday'].value_counts().plot(kind='bar')
plt.xlabel("Day of Week")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Day of Week")
plt.tight_layout()
plt.savefig("outputs/accidents_by_weekday.png")
plt.close()

print("Graph saved: accidents_by_weekday.png")


# -----------------------------
# GRAPH 3: ACCIDENTS BY MONTH
# -----------------------------
plt.figure(figsize=(10, 5))
df['Month'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.title("Accidents by Month")
plt.tight_layout()
plt.savefig("outputs/accidents_by_month.png")
plt.close()

print("Graph saved: accidents_by_month.png")

# -----------------------------
# GRAPH 4: ACCIDENT SEVERITY
# -----------------------------
plt.figure(figsize=(8, 5))
df['Severity'].value_counts().sort_index().plot(kind='bar')
plt.xlabel("Severity Level")
plt.ylabel("Number of Accidents")
plt.title("Accident Severity Distribution")
plt.tight_layout()
plt.savefig("outputs/accident_severity.png")
plt.close()

print("Graph saved: accident_severity.png")

# -----------------------------
# GRAPH 5: Weather Condition Analysis
# -----------------------------
top_weather = df["Weather_Condition"].value_counts().head(10)

plt.figure()
top_weather.plot(kind="bar")
plt.title("Top 10 Weather Conditions During Accidents")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/weather_conditions.png")
plt.close()

# -----------------------------
#Road Condition Analysis
# -----------------------------
road_features = {
    "Junction": df["Junction"].sum(),
    "Traffic Signal": df["Traffic_Signal"].sum(),
    "Crossing": df["Crossing"].sum(),
    "Roundabout": df["Roundabout"].sum()
}

road_df = pd.Series(road_features)

plt.figure()
road_df.plot(kind="bar")
plt.title("Accidents by Road Features")
plt.xlabel("Road Feature")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.savefig("outputs/road_features.png")
plt.close()

# -----------------------------
# OUTLIER ANALYSIS & HANDLING
# -----------------------------

# ----- Visibility Outliers -----
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Visibility(mi)"])
plt.title("Visibility Outliers")
plt.tight_layout()
plt.savefig("outputs/visibility_outliers.png")
plt.close()

# Cap extreme visibility values (99th percentile)
visibility_upper = df["Visibility(mi)"].quantile(0.99)
df["Visibility_capped"] = df["Visibility(mi)"].clip(upper=visibility_upper)

# ----- Temperature Outliers -----
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Temperature(F)"])
plt.title("Temperature Outliers")
plt.tight_layout()
plt.savefig("outputs/temperature_outliers.png")
plt.close()

# Cap extreme temperature values
temp_lower = df["Temperature(F)"].quantile(0.01)
temp_upper = df["Temperature(F)"].quantile(0.99)

df["Temperature_capped"] = df["Temperature(F)"].clip(
    lower=temp_lower,
    upper=temp_upper
)

# -----------------------------
#Visibility
# -----------------------------

plt.figure(figsize=(10, 5))
df = df[df["Visibility(mi)"] <= 20]
sns.histplot(df["Visibility_capped"], bins=30, kde=True)

plt.title("Visibility Distribution During Accidents", fontsize=14)
plt.xlabel("Visibility (miles)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/visibility_histogram.png")
plt.close()

# -----------------------------
#Sevirity:Pie chart
# -----------------------------
severity_counts = df["Severity"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(
    severity_counts,
    labels=severity_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Percentage Distribution of Accident Severity", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/severity_pie.png")
plt.close()

## -----------------------------
#Temperature
# -----------------------------
plt.figure(figsize=(10, 6))
sns.histplot(df["Temperature_capped"].dropna(), bins=30, kde=True)

plt.title("Temperature Distribution During Accidents", fontsize=14)
plt.xlabel("Temperature (°F)", fontsize=12)
plt.ylabel("Number of Accidents", fontsize=12)

plt.tight_layout()
plt.savefig("outputs/temperature_histogram.png")
plt.close()

# ---------- BIVARIATE ANALYSIS ----------
# Severity vs Visibility
plt.figure(figsize=(8, 6))

sns.boxplot(
    x="Severity",
    y="Visibility(mi)",
    hue="Severity",
    data=df,
    palette="Set2",
    legend=False
)

plt.title("Accident Severity vs Visibility", fontsize=14)
plt.xlabel("Severity Level")
plt.ylabel("Visibility (miles)")

plt.tight_layout()
plt.savefig("outputs/severity_vs_visibility.png")
plt.close()


# Severity vs Road conditions
road_surface = df.copy()

road_surface["Road_Surface"] = road_surface["Weather_Condition"].apply(
    lambda x: "Wet" if "Rain" in str(x) or "Snow" in str(x) else "Dry"
)

surface_severity = road_surface.groupby("Road_Surface")["Severity"].mean()

plt.figure(figsize=(6,4))
sns.barplot(
    x=surface_severity.index,
    y=surface_severity.values,
    hue=surface_severity.index,
    palette="coolwarm",
    legend=False
)
plt.title("Severity vs Road Surface Condition")
plt.xlabel("Road Surface")
plt.ylabel("Average Severity")
plt.tight_layout()
plt.savefig("outputs/severity_vs_road_surface.png")
plt.close()


# Severity vs Weather
weather_severity = df.groupby("Weather_Condition")["Severity"].mean().sort_values(ascending=False).head(10)

sns.barplot(
    x=weather_severity.values,
    y=weather_severity.index,
    hue=weather_severity.index,
    palette="viridis",
    legend=False
)

plt.title("Average Severity by Weather Condition")
plt.savefig("outputs/severity_vs_weather.png")
plt.close()


#Severity vs Traffic Congestion
df["Traffic_Congestion_Score"] = (
    df["Traffic_Signal"].astype(int) +
    df["Junction"].astype(int) +
    df["Crossing"].astype(int) +
    df["Stop"].astype(int)
)
plt.figure(figsize=(8,5))
sns.boxplot(
    x="Severity",
    y="Traffic_Congestion_Score",
    hue="Severity",
    data=df,
    palette="Set3",
    legend=False
)

plt.title("Accident Severity vs Traffic Congestion")
plt.xlabel("Severity Level")
plt.ylabel("Traffic Congestion Score")
plt.tight_layout()
plt.savefig("outputs/severity_vs_congestion.png")
plt.close()


#CORRELATION HEATMAP
corr_cols = [
    "Severity",
    "Visibility(mi)",
    "Traffic_Congestion_Score",
    "Precipitation(in)"
]

plt.figure(figsize=(6,4))
sns.heatmap(
    df[corr_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap: Severity vs Key Factors")
plt.tight_layout()
plt.savefig("outputs/severity_correlation_heatmap.png")
plt.close()

# =============================
# WEEK 6: HYPOTHESIS TESTING
# =============================
from scipy.stats import ttest_ind

# H1: Severity during Rain/Fog vs Clear
rainy = df[df["Weather_Condition"].str.contains("Rain|Fog", na=False)]["Severity"]
clear = df[df["Weather_Condition"] == "Clear"]["Severity"]

t_stat, p_value = ttest_ind(rainy, clear, equal_var=False)

print("\nHypothesis Test 1: Rain/Fog vs Clear")
print("p-value:", p_value)

# H2: Visibility vs Severity
low_visibility = df[df["Visibility(mi)"] < 3]["Severity"]
high_visibility = df[df["Visibility(mi)"] >= 10]["Severity"]

t_stat2, p_value2 = ttest_ind(low_visibility, high_visibility, equal_var=False)

print("\nHypothesis Test 2: Visibility impact on Severity")
print("p-value:", p_value2)
