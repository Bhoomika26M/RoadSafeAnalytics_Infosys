import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("US_Accidents_March23.csv")  # use your actual file

# 1) Clean and convert Start_Time
# remove the .000000000 part if present
df["Start_Time_clean"] = df["Start_Time"].astype(str).str.split(".").str[0]

df["datetime"] = pd.to_datetime(df["Start_Time_clean"],
                                format="%Y-%m-%d %H:%M:%S",
                                errors="coerce")
df = df.dropna(subset=["datetime"])  # drop rows that still failed
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day_name()

# 2) Replace column names below with your EXACT names
severity_col = "Severity"            # change if different
weather_col = "Weather_Condition"    # change if different

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

df[severity_col].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[0, 0])
axes[0, 0].set_title("Severity")

axes[0, 1].hist(df["hour"], bins=24, edgecolor="black")
axes[0, 1].set_title("By Hour")

df["day"].value_counts().plot.bar(ax=axes[1, 0])
axes[1, 0].set_title("By Day")
axes[1, 0].tick_params(axis="x", rotation=45)

df[weather_col].value_counts().head(5).plot.barh(ax=axes[1, 1])
axes[1, 1].set_title("Top Weather")

plt.tight_layout()
plt.show()

# Bivariate: severity vs weather
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x=weather_col, y=severity_col)
plt.xticks(rotation=45)
plt.title("Severity by Weather")
plt.tight_layout()
plt.show()
