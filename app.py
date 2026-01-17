import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------
# Global Plot Font Settings (Clean Look)
# ------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

# ------------------------------------
# Streamlit Page Config
# ------------------------------------
st.set_page_config(
    page_title="RoadSafe Analytics",
    layout="wide"
)

# ------------------------------------
# Custom CSS for Font Control (OPTION 4)
# ------------------------------------
st.markdown("""
<style>
h1 { font-size: 36px; }
h2 { font-size: 26px; }
h3 { font-size: 20px; }
p  { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------
# Title Section
# ------------------------------------
st.title("🚦 RoadSafe Analytics Dashboard")
st.write("Exploratory Data Analysis of US Road Accidents")

# ------------------------------------
# Load Dataset
# ------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset/US_Accidents_March23.csv")

df = load_data()

# ------------------------------------
# Dataset Preview
# ------------------------------------
st.markdown("### 📄 Dataset Preview")
st.dataframe(df.head(50), use_container_width=True)

# ------------------------------------
# Accident Severity Distribution
# ------------------------------------
st.markdown("### 🚨 Accident Severity Distribution")

severity_counts = df['Severity'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x=severity_counts.index,
    y=severity_counts.values,
    palette="Reds",
    ax=ax
)

ax.set_xlabel("Severity Level (1 = Low, 4 = High)")
ax.set_ylabel("Number of Accidents")
ax.set_title("Distribution of Accident Severity")

st.pyplot(fig)

st.info(
    "Most accidents fall under Severity levels 2 and 3, "
    "indicating that moderate-impact crashes are the most common."
)

# ------------------------------------
# Accidents by Hour of Day
# ------------------------------------
st.markdown("### ⏰ Accidents by Hour of Day")

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Hour'] = df['Start_Time'].dt.hour

hour_counts = df['Hour'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7,4))
sns.barplot(
    x=hour_counts.index,
    y=hour_counts.values,
    color="steelblue",
    ax=ax
)

ax.set_xlabel("Hour of Day (0–23)")
ax.set_ylabel("Number of Accidents")
ax.set_title("Accidents Distribution by Hour")

st.pyplot(fig)

st.info(
    "Accident frequency peaks during morning and evening rush hours, "
    "highlighting traffic congestion as a major contributing factor."
)

# ------------------------------------
# Top 5 Accident-Prone States
# ------------------------------------
st.markdown("### 📍 Top 5 Accident-Prone States")

top_states = df['State'].value_counts().head(5)

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(
    x=top_states.values,
    y=top_states.index,
    color="tomato",
    ax=ax
)

ax.set_xlabel("Number of Accidents")
ax.set_ylabel("State")
ax.set_title("Top 5 States with Highest Accident Counts")

st.pyplot(fig)

st.info(
    "A small number of large states account for a significant proportion "
    "of total accidents, likely due to higher population and traffic density."
)

# ------------------------------------
# Geospatial Accident Hotspots
# ------------------------------------
st.markdown("### 🌍 Accident Hotspots Across the US")

geo_sample = df[['Start_Lat', 'Start_Lng']].dropna().sample(50000, random_state=42)

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(
    geo_sample['Start_Lng'],
    geo_sample['Start_Lat'],
    s=1,
    alpha=0.3
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Accident Hotspots (Latitude vs Longitude)")

st.pyplot(fig)

st.info(
    "Accident hotspots are heavily concentrated in urban regions and "
    "along major highway corridors across the United States."
)

# ------------------------------------
# Key Takeaways
# ------------------------------------
st.markdown("### 📌 Key Takeaways")

st.markdown("""
- Most road accidents are of **moderate severity**, rather than extreme.
- **Time of day** plays a critical role, with peaks during rush hours.
- Accident distribution is **geographically uneven**, concentrated in a few states.
- Urban areas and highways emerge as major **accident hotspots**.
- Human activity and traffic patterns influence accidents more than weather factors.
""")
