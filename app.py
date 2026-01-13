import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="RoadSafe Analytics Dashboard",
    layout="wide"
)

# -------------------------------------------------
# TITLE & INTRODUCTION
# -------------------------------------------------
st.title("🚦 RoadSafe Analytics – Road Accident Dashboard")
st.markdown("""
This interactive dashboard presents **Exploratory Data Analysis (EDA)**  
on **US Road Accident Data** to understand accident severity, time trends,  
weather impact, visibility conditions, and location-based patterns.
""")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_sample.csv")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.day_name()
    return df

df = load_data()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("🔎 Filter Data")

state_filter = st.sidebar.multiselect(
    "Select State(s)",
    options=sorted(df['State'].unique()),
    default=sorted(df['State'].unique())[:5]
)

weather_filter = st.sidebar.multiselect(
    "Select Weather Condition(s)",
    options=sorted(df['Weather_Condition'].unique()),
    default=sorted(df['Weather_Condition'].unique())[:5]
)

filtered_df = df[
    (df['State'].isin(state_filter)) &
    (df['Weather_Condition'].isin(weather_filter))
]

# -------------------------------------------------
# OVERVIEW METRICS
# -------------------------------------------------
st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Accidents", len(filtered_df))
col2.metric("Average Severity", round(filtered_df['Severity'].mean(), 2))
col3.metric("Number of States", filtered_df['State'].nunique())

st.markdown("""
**Severity Levels** range from **1 (least severe)** to **4 (most severe)**.
""")

# -------------------------------------------------
# SEVERITY DISTRIBUTION (UNIVARIATE)
# -------------------------------------------------
st.header("🚑 Accident Severity Distribution")

fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x='Severity', ax=ax1)
ax1.set_xlabel("Severity Level")
ax1.set_ylabel("Number of Accidents")
st.pyplot(fig1)

st.markdown("""
This chart shows how accidents are distributed across different severity levels.
""")

# -------------------------------------------------
# TIME-BASED ANALYSIS
# -------------------------------------------------
st.header("⏰ Time-Based Accident Analysis")

fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['Hour'], bins=24, kde=True, ax=ax2)
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Accident Count")
st.pyplot(fig2)

st.markdown("""
This plot answers the question:  
**At what time of day do most accidents occur?**
""")

# -------------------------------------------------
# WEATHER VS SEVERITY (BIVARIATE)
# -------------------------------------------------
st.header("🌦️ Weather Conditions vs Accident Severity")

fig3, ax3 = plt.subplots(figsize=(10,4))
sns.boxplot(
    data=filtered_df,
    x='Weather_Condition',
    y='Severity',
    ax=ax3
)
plt.xticks(rotation=45)
st.pyplot(fig3)

st.markdown("""
This plot shows how accident severity varies under different weather conditions.
""")

# -------------------------------------------------
# VISIBILITY VS SEVERITY
# -------------------------------------------------
st.header("👁️ Visibility vs Accident Severity")

fig4, ax4 = plt.subplots()
sns.scatterplot(
    data=filtered_df,
    x='Visibility(mi)',
    y='Severity',
    alpha=0.3,
    ax=ax4
)
st.pyplot(fig4)

st.markdown("""
Lower visibility conditions often correlate with higher accident severity.
""")

# -------------------------------------------------
# LOCATION-BASED ANALYSIS
# -------------------------------------------------
st.header("📍 Location-Based Analysis")

top_states = filtered_df['State'].value_counts().head(5)

fig5, ax5 = plt.subplots()
top_states.plot(kind='bar', ax=ax5)
ax5.set_xlabel("State")
ax5.set_ylabel("Number of Accidents")
st.pyplot(fig5)

st.markdown("""
This chart highlights the **top 5 accident-prone states**.
""")

# -------------------------------------------------
# FINAL INSIGHTS
# -------------------------------------------------
st.header("📌 Key Insights & Conclusions")

st.markdown("""
- 🚗 Most accidents occur during peak traffic hours  
- 🌧️ Rain and fog increase accident severity  
- 👁️ Poor visibility is associated with higher severity  
- 🗺️ Certain states consistently show higher accident counts  
""")

st.success("✅ Streamlit Dashboard Loaded Successfully")
