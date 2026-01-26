import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Road Safe Analytics Dashboard",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🚦 Road Safe Analytics Dashboard")
st.markdown("**Comprehensive dashboard covering all milestones**")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/US_Accidents_Cleaned.csv", nrows=1000000)

df = load_data()

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("🔎 Filters")

severity_filter = st.sidebar.multiselect(
    "Severity (Impact Level)",
    sorted(df["Severity"].unique()),
    default=sorted(df["Severity"].unique())
)

weekday_filter = st.sidebar.multiselect(
    "Weekday",
    df["Weekday"].unique(),
    default=df["Weekday"].unique()
)

city_filter = st.sidebar.multiselect(
    "City",
    sorted(df["City"].dropna().unique())
)

hour_range = st.sidebar.slider(
    "Hour Range (Rush-hour Analysis)",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

# --------------------------------------------------
# Apply Filters
# --------------------------------------------------
filtered_df = df[
    (df["Severity"].isin(severity_filter)) &
    (df["Weekday"].isin(weekday_filter)) &
    (df["Hour"] >= hour_range[0]) &
    (df["Hour"] <= hour_range[1])
]

if city_filter:
    filtered_df = filtered_df[filtered_df["City"].isin(city_filter)]

# --------------------------------------------------
# KPIs
# --------------------------------------------------
st.subheader("📌 Key Statistics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Accidents", len(filtered_df))
c2.metric("Average Severity", round(filtered_df["Severity"].mean(), 2))
c3.metric("Weather Types", filtered_df["Weather_Condition"].nunique())
c4.metric("Cities Selected", filtered_df["City"].nunique())

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 EDA",
    "⏱ Time Analysis",
    "⚠ Severity Analysis",
    "🌍 Geo & Hypothesis",
    "📊 Visualization & Interpretation",
    "🗺 Interactive Risk Map",
    "❓ Help"
])

# ==================================================
# TAB 1 – EDA (Week 3)
# ==================================================
with tab1:
    st.subheader("Accident Severity Distribution")
    fig, ax = plt.subplots()
    filtered_df["Severity"].value_counts().sort_index().plot(kind="bar", ax=ax)
    st.pyplot(fig)


    st.subheader("Accident Severity Share (Pie Chart)")
    weekday_counts = filtered_df["Weekday"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(
        weekday_counts,
        labels=weekday_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Accident Distribution by Weekday")
    st.pyplot(fig)


    st.subheader("Top Weather Conditions")
    fig, ax = plt.subplots(figsize=(10,4))
    filtered_df["Weather_Condition"].value_counts().head(10).plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ==================================================
# TAB 2 – TIME ANALYSIS
# ==================================================
with tab2:
    st.subheader("Accidents by Hour")
    fig, ax = plt.subplots()
    filtered_df["Hour"].value_counts().sort_index().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("Accidents by Month")
    fig, ax = plt.subplots()
    filtered_df["Month"].value_counts().sort_index().plot(kind="line", marker="o", ax=ax)
    st.pyplot(fig)

# ==================================================
# TAB 3 – SEVERITY ANALYSIS (Week 4)
# ==================================================
with tab3:
    st.subheader("Correlation Heatmap")

    corr_cols = [
        "Severity",
        "Visibility(mi)",
        "Temperature(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Wind_Speed(mph)"
    ]

    corr_df = filtered_df[corr_cols].dropna()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Severity vs Visibility (Box Plot)")
    vis_df = filtered_df[filtered_df["Visibility(mi)"] < 10]
    if len(vis_df) < 50:
        vis_df = filtered_df

    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(x="Severity", y="Visibility(mi)", data=vis_df, ax=ax)
    st.pyplot(fig)

    st.subheader("Severity vs Visibility (Violin Plot)")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.violinplot(
        x="Severity",
        y="Visibility(mi)",
        data=vis_df,
        inner="quartile",
        ax=ax
    )
    st.pyplot(fig)

# ==================================================
# TAB 4 – GEO & HYPOTHESIS (Week 5 & 6)
# ==================================================
with tab4:
    st.subheader("Accident Locations (Dark Scatter Plot)")

    geo_df = filtered_df.sample(min(20000, len(filtered_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(
        geo_df["Start_Lng"],
        geo_df["Start_Lat"],
        s=2,
        c="darkred",
        alpha=0.6
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

    st.subheader("Hypothesis Testing")
    peak_hour = filtered_df["Hour"].value_counts().idxmax()
    st.success(f"Peak accident hour: {peak_hour}:00")

    corr_val = filtered_df["Severity"].corr(filtered_df["Visibility(mi)"])
    st.info(f"Visibility vs Severity Correlation: {corr_val:.2f}")

# ==================================================
# TAB 5 – VISUALIZATION & INTERPRETATION (Week 7)
# ==================================================
with tab5:
    st.header("📊 Visualization & Interpretation")

    fig, ax = plt.subplots()
    filtered_df["Hour"].value_counts().sort_index().plot(kind="line", marker="o", ax=ax)
    st.pyplot(fig)
    st.success("Accidents peak during morning and evening rush hours.")

    weather_sev = (
        filtered_df.groupby("Weather_Condition")["Severity"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
    )

    fig, ax = plt.subplots()
    weather_sev.plot(kind="bar", ax=ax)
    st.pyplot(fig)
    st.success("Bad weather conditions increase accident severity.")

    st.info("Urban areas and highways require targeted safety measures.")

# ==================================================
# TAB 6 – INTERACTIVE RISK MAP (STABLE)
# ==================================================
with tab6:
    st.subheader("🗺 Interactive Accident Danger Hotspots")

    map_df = filtered_df[
        ["Start_Lat", "Start_Lng", "Severity"]
    ].dropna().sample(min(50000, len(filtered_df)), random_state=42)

    center_lat = map_df["Start_Lat"].mean()
    center_lng = map_df["Start_Lng"].mean()

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=4,
        tiles="OpenStreetMap"
    )

    heat_data = [
        [row["Start_Lat"], row["Start_Lng"], row["Severity"]]
        for _, row in map_df.iterrows()
    ]

    HeatMap(
        heat_data,
        radius=8,
        blur=10,
        max_zoom=6
    ).add_to(m)

    st_folium(m, width=1000, height=500)

# ==================================================
# TAB 7 – HELP
# ==================================================
with tab7:
    st.header("❓ Dashboard Help & User Guide")

    st.markdown("""
    ### 🔎 How to Use Filters
    - **Severity**: Filters accidents by impact level  
      *(1 = Minor, 4 = Severe/Fatal)*  
    - **Weekday**: Analyze accident patterns across days  
    - **City**: Focus on location-specific accident risk  
    - **Hour Range**: Identify rush-hour and off-peak accidents  

    Any filter selection dynamically updates **all charts and maps**.

    ---
    ### 📊 Understanding the Visualizations
    - **Bar Charts**: Show frequency of accidents  
    - **Pie Chart**: Shows proportional distribution (severity share)  
    - **Box & Violin Plots**: Show distribution and variability  
    - **Correlation Heatmap**: Shows linear relationships  
    - **Scatter Plot**: Displays exact accident locations  
    - **Interactive Map**: Highlights accident hotspots  

    ---
    ### 🧠 Key Analytical Notes
    - Correlation values near zero indicate weak *linear* relationships  
    - Some plots may appear sparse if filtered data is limited  
    - High accident counts can occur even under good visibility  

    ---
    ### 🚀 Performance Optimization
    - Large datasets are sampled to reduce latency  
    - Heatmaps are used instead of individual markers  
    - Filters are applied before visualization rendering  

    ---
    ### 🎯 Dashboard Purpose
    This dashboard supports **data-driven road safety analysis**, helping
    identify high-risk times, locations, and conditions for targeted interventions.
    """)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown("🚦 RoadSafe Analytics | Final Dashboard")
