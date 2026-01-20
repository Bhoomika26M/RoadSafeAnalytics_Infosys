import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RoadSafe Analytics Dashboard",
    layout="wide"
)

sns.set_theme(style="darkgrid")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/US_Accidents_March23.csv")

    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df = df.dropna(subset=['Start_Time'])

    df['Date'] = df['Start_Time'].dt.normalize()
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.day_name()

    df['Time_Of_Day'] = df['Hour'].apply(
        lambda x: "Day" if 6 <= x < 18 else "Night"
    )

    return df

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🔍 Filters")

states = st.sidebar.multiselect(
    "Select States",
    sorted(df['State'].dropna().unique()),
    default=["CA", "TX", "FL"]
)

severity = st.sidebar.multiselect(
    "Select Severity",
    sorted(df['Severity'].dropna().unique()),
    default=[2, 3]
)

weather = st.sidebar.multiselect(
    "Select Weather",
    sorted(df['Weather_Condition'].dropna().unique()),
    default=df['Weather_Condition'].dropna().unique()[:5]
)

time_of_day = st.sidebar.radio(
    "Time of Day",
    ["All", "Day", "Night"]
)

min_date = df['Date'].min()
max_date = df['Date'].max()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# ---------------- APPLY FILTERS ----------------
filtered_df = df[
    (df['State'].isin(states)) &
    (df['Severity'].isin(severity)) &
    (df['Weather_Condition'].isin(weather)) &
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1]))
]

if time_of_day != "All":
    filtered_df = filtered_df[filtered_df['Time_Of_Day'] == time_of_day]

st.sidebar.success(f"Records: {len(filtered_df)}")

# ---------------- TITLE ----------------
st.title("🚦 RoadSafe Analytics – Interactive Dashboard")
st.markdown("US Road Accident Analysis with Dynamic Filters")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "⏰ Time Analysis", "🌍 Geospatial", "🔥 Insights"]
)

# =====================================================
# TAB 1: OVERVIEW (4 PLOTS)
# =====================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Severity Distribution")
        sev_cnt = filtered_df['Severity'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(x=sev_cnt.index, y=sev_cnt.values, palette="Reds", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Top Accident-Prone States")
        top_states = filtered_df['State'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(x=top_states.values, y=top_states.index, palette="Blues", ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Accidents by Weather")
        weather_cnt = filtered_df['Weather_Condition'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(x=weather_cnt.values, y=weather_cnt.index, palette="coolwarm", ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("Day vs Night Accidents")
        tod_cnt = filtered_df['Time_Of_Day'].value_counts()
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(x=tod_cnt.index, y=tod_cnt.values, palette="Set2", ax=ax)
        st.pyplot(fig)

# =====================================================
# TAB 2: TIME ANALYSIS (4 PLOTS)
# =====================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accidents by Hour")
        hour_cnt = filtered_df['Hour'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4,3))
        sns.lineplot(x=hour_cnt.index, y=hour_cnt.values, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Accidents by Weekday")
        day_cnt = filtered_df['Weekday'].value_counts()
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(x=day_cnt.values, y=day_cnt.index, palette="viridis", ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Severity vs Time of Day")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.countplot(
            data=filtered_df,
            x="Severity",
            hue="Time_Of_Day",
            palette="Set1",
            ax=ax
        )
        st.pyplot(fig)

    with col4:
        st.subheader("Hourly Severity Heatmap")
        heat = pd.crosstab(filtered_df['Hour'], filtered_df['Severity'])
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(heat, cmap="YlOrRd", ax=ax)
        st.pyplot(fig)

# =====================================================
# TAB 3: GEOSPATIAL
# =====================================================
with tab3:
    st.subheader("Accident Hotspots")

    geo_df = filtered_df[['Start_Lat', 'Start_Lng']].dropna()
    geo_df = geo_df.sample(min(40000, len(geo_df)), random_state=42)

    st.map(
        geo_df.rename(columns={
            "Start_Lat": "lat",
            "Start_Lng": "lon"
        })
    )

# =====================================================
# TAB 4: INSIGHTS
# =====================================================
with tab4:
    st.subheader("Correlation Heatmap")

    corr_cols = [
        'Severity',
        'Distance(mi)',
        'Temperature(F)',
        'Humidity(%)',
        'Visibility(mi)'
    ]

    corr_df = filtered_df[corr_cols].dropna()

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(
        corr_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    st.pyplot(fig)

    st.markdown("""
    **Key Insights**
    - Majority accidents occur during **daytime** but night accidents tend to be severe
    - **Rush hours** show highest accident density
    - Weather conditions alone do not strongly impact severity
    - Urban regions dominate accident hotspots
    """)

