import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

st.set_page_config(page_title="RoadSafe Analytics", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_parquet("US_Accidents_clean.parquet")

df = load_data()

st.title("🚦 RoadSafe Analytics Dashboard")
st.write("US Road Accidents – Geospatial & Severity Analysis")

st.write("Dataset shape:", df.shape)

# -----------------------------
# SECTION 1: Time-Based Analysis
# -----------------------------
st.header("⏰ Time-Based Accident Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Accidents by Hour")
    fig, ax = plt.subplots()
    df["Hour"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Accidents")
    st.pyplot(fig)

with col2:
    st.subheader("Accidents by Day of Week")
    fig, ax = plt.subplots()
    df["Weekday"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Day")
    ax.set_ylabel("Accidents")
    st.pyplot(fig)

st.subheader("Accidents by Month")
fig, ax = plt.subplots()
df["Month"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Accidents")
st.pyplot(fig)

# -----------------------------
# SECTION 2: Severity Analysis
# -----------------------------
st.header("🚑 Accident Severity")

fig, ax = plt.subplots()
df["Severity"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Severity Level")
ax.set_ylabel("Count")
st.pyplot(fig)

# -----------------------------
# SECTION 3: Geospatial Analysis
# -----------------------------
st.header("📍 Accident Hotspots")

sample_df = df.sample(10000, random_state=42)
map_df = sample_df.rename(columns={
    "Start_Lat": "lat",
    "Start_Lng": "lon"
})

st.map(map_df[["lat", "lon"]])


top_states = df["State"].value_counts().head(5)
st.subheader("Top 5 Accident-Prone States")
st.bar_chart(top_states)

# -----------------------------
# SECTION 4: Weather & Visibility
# -----------------------------
st.header("🌦 Weather & Visibility Impact")

fig, ax = plt.subplots()
sns.boxplot(
    x="Severity",
    y="Visibility(mi)",
    data=df,
    hue="Severity",
    legend=False,
    ax=ax
)
ax.set_title("Severity vs Visibility")
st.pyplot(fig)

top_weather = df.groupby("Weather_Condition")["Severity"].mean().sort_values(ascending=False).head(10)
st.subheader("Average Severity by Weather Condition")
st.bar_chart(top_weather)

# -----------------------------
# SECTION 5: Hypothesis Testing (Results)
# -----------------------------
st.header("📊 Hypothesis Testing Results")

st.markdown("""
### Hypothesis 1  
**Are accidents more severe during rain or fog?**

✔ p-value < 0.05  
➡ We reject the null hypothesis  
➡ Accidents during rain/fog are statistically more severe

---

### Hypothesis 2  
**Is visibility related to accident severity?**

✔ p-value < 0.05  
➡ Strong negative correlation  
➡ Lower visibility → higher severity
""")

st.success("Milestone 3 analysis completed successfully!")
