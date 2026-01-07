import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set_palette("Set2")
plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs("outputs", exist_ok=True)
df = pd.read_parquet("US_Accidents_clean.parquet")

print("Data loaded")
print(df.shape)

#GEOSPATIAL ANALYSIS
#Accident Hotspots (Latitude vs Longitude)

plt.figure(figsize=(8,6))

plt.scatter(
    df["Start_Lng"],
    df["Start_Lat"],
    s=1,
    alpha=0.3
)

plt.title("Accident Hotspots (Geospatial View)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig("outputs/accident_hotspots.png")
plt.close()

#Top 5 Accident-Prone States

top_states = df["State"].value_counts().head(5)

plt.figure(figsize=(6,4))
top_states.plot(kind="bar")

plt.title("Top 5 Accident-Prone States")
plt.xlabel("State")
plt.ylabel("Number of Accidents")

plt.tight_layout()
plt.savefig("outputs/top_states.png")
plt.close()

#Top 5 Accident-Prone Cities

top_cities = df["City"].value_counts().head(5)

plt.figure(figsize=(6,4))
top_cities.plot(kind="bar")

plt.title("Top 5 Accident-Prone Cities")
plt.xlabel("City")
plt.ylabel("Number of Accidents")

plt.tight_layout()
plt.savefig("outputs/top_cities.png")
plt.close()

#INSIGHT EXTRACTION & HYPOTHESIS TESTING
#Are accidents more severe in rain or fog?
weather_severity = (
    df[df["Weather_Condition"].isin(["Rain", "Fog"])]
    .groupby("Weather_Condition")["Severity"]
    .mean()
)

plt.figure(figsize=(5,4))
weather_severity.plot(kind="bar")

plt.title("Average Severity in Rain vs Fog")
plt.xlabel("Weather Condition")
plt.ylabel("Average Severity")

plt.tight_layout()
plt.savefig("outputs/weather_vs_severity.png")
plt.close()
