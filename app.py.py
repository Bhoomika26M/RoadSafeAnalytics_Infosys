import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px # New library for interactive charts

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RoadSafe Analytics Pro",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA FUNCTION
# ==========================================
@st.cache_data
def load_data():
    file_path = 'cleaned_accident_data.csv'
    try:
        # Loading 500k rows
        df = pd.read_csv(file_path, nrows=500000)
        
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        df['Year'] = df['Start_Time'].dt.year
        df['Month'] = df['Start_Time'].dt.month_name()
        df['DayOfWeek'] = df['Start_Time'].dt.day_name()
        df['Hour'] = df['Start_Time'].dt.hour
        
        # Handling missing values for visualization
        if 'Visibility(mi)' not in df.columns: df['Visibility(mi)'] = 10.0
        if 'Sunrise_Sunset' not in df.columns: df['Sunrise_Sunset'] = 'Day'
        if 'Humidity(%)' not in df.columns: df['Humidity(%)'] = 50.0
        if 'Temperature(F)' not in df.columns: df['Temperature(F)'] = 70.0
            
        return df
    except FileNotFoundError:
        st.error("File 'cleaned_accident_data.csv' not found.")
        return None

df = load_data()

# ==========================================
# 3. SIDEBAR FILTERS
# ==========================================
if df is not None:
    st.sidebar.header("🔍 Advanced Filters")
    
    # Date Range
    min_date = df['Start_Time'].min().date()
    max_date = df['Start_Time'].max().date()
    date_range = st.sidebar.date_input("📅 Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # State Filter
    all_states = sorted(df['State'].astype(str).unique())
    selected_states = st.sidebar.multiselect("🗺️ Select State(s)", all_states, default=all_states[:3])

    # Weather Filter
    all_weather = df['Weather_Condition'].dropna().unique()
    selected_weather = st.sidebar.multiselect("🌤️ Weather Condition", all_weather)

    # Severity Filter
    all_severity = sorted(df['Severity'].unique())
    selected_severity = st.sidebar.multiselect("⚠️ Severity Level", all_severity, default=all_severity)

    # --- APPLY FILTERS ---
    filtered_df = df.copy()
    if len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = filtered_df[(filtered_df['Start_Time'].dt.date >= start_d) & (filtered_df['Start_Time'].dt.date <= end_d)]
    
    if selected_states:
        filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
    if selected_weather:
        filtered_df = filtered_df[filtered_df['Weather_Condition'].isin(selected_weather)]
    if selected_severity:
        filtered_df = filtered_df[filtered_df['Severity'].isin(selected_severity)]

    # ==========================================
    # 4. MAIN DASHBOARD
    # ==========================================
    st.title("🚦 RoadSafe Analytics: US Accident Dashboard")
    st.markdown("### 📊 Comprehensive Traffic Safety Analysis")
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Accidents", f"{len(filtered_df):,}")
    c2.metric("Top City", filtered_df['City'].mode()[0] if not filtered_df.empty else "N/A")
    c3.metric("Avg Severity", f"{filtered_df['Severity'].mean():.2f}")
    c4.metric("Avg Visibility", f"{filtered_df['Visibility(mi)'].mean():.1f} mi")
    st.markdown("---")

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Univariate Analysis", "📈 Bivariate & Advanced", "🗺️ Map & Data"])

    # --- TAB 1: UNIVARIATE ---
    with tab1:
        st.subheader("General Trends & Distributions")
        
        # Row 1: Hour & Weekday
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("**1. Hourly Accident Trend**")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(filtered_df['Hour'], bins=24, kde=True, color='skyblue', ax=ax)
            st.pyplot(fig)
        with r1c2:
            st.markdown("**2. Weekly Accident Trend**")
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x='DayOfWeek', data=filtered_df, order=days_order, palette='viridis', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Row 2: Top Cities & Top States (NEW)
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("**3. Top 10 Cities**")
            city_counts = filtered_df['City'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=city_counts.values, y=city_counts.index, palette='magma', ax=ax)
            st.pyplot(fig)
        with r2c2:
            st.markdown("**4. Top 10 States (NEW)**")
            state_counts = filtered_df['State'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=state_counts.index, y=state_counts.values, palette='coolwarm', ax=ax)
            st.pyplot(fig)

        # Row 3: Severity Pie Chart (NEW)
        st.markdown("---")
        r3c1, r3c2 = st.columns([1, 2])
        with r3c1:
            st.markdown("**5. Severity Distribution (Percentage)**")
            sev_counts = filtered_df['Severity'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sev_counts, labels=sev_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            st.pyplot(fig)
        with r3c2:
             st.markdown("**6. Monthly Trend**")
             months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
             existing_months = [m for m in months_order if m in filtered_df['Month'].unique()]
             fig, ax = plt.subplots(figsize=(10, 4))
             sns.countplot(x='Month', data=filtered_df, order=existing_months, palette='rocket', ax=ax)
             plt.xticks(rotation=45)
             st.pyplot(fig)

    # --- TAB 2: BIVARIATE & ADVANCED ---
    with tab2:
        st.subheader("Deep Dive Analysis")

        # Row 1: Correlation Heatmap (NEW - Very Important)
        st.markdown("### 🔥 Correlation Heatmap: Weather vs Accidents")
        st.write("This chart shows relationships between numerical variables.")
        corr_cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']
        # Check if cols exist
        valid_cols = [c for c in corr_cols if c in filtered_df.columns]
        if len(valid_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(filtered_df[valid_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numerical columns for heatmap.")

        st.markdown("---")
        
        # Row 2: Day vs Night & Temperature Box Plot
        r2c1, r2c2 = st.columns(2)
        
        with r2c1:
            st.markdown("**🌞 Day vs 🌙 Night Accidents**")
            if 'Sunrise_Sunset' in filtered_df.columns:
                day_counts = filtered_df['Sunrise_Sunset'].value_counts()
                fig, ax = plt.subplots()
                # Donut Chart
                ax.pie(day_counts, labels=day_counts.index, autopct='%1.1f%%', colors=['gold', 'black'], wedgeprops=dict(width=0.3))
                st.pyplot(fig)
        
        with r2c2:
            st.markdown("**🌡️ Temperature Impact on Severity**")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x='Severity', y='Temperature(F)', data=filtered_df, palette='Set2', ax=ax)
            ax.set_title("Temperature Distribution by Severity")
            st.pyplot(fig)

    # --- TAB 3: MAP & DATA ---
    with tab3:
        st.markdown("### 🗺️ Geographic Hotspots")
        if not filtered_df.empty:
            map_data = filtered_df[['Start_Lat', 'Start_Lng']].rename(columns={'Start_Lat': 'lat', 'Start_Lng': 'lon'}).dropna()
            st.map(map_data.sample(min(len(map_data), 15000)))
        
        st.markdown("### 📋 Filtered Data")
        st.dataframe(filtered_df.head(100))
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "filtered_data.csv", "text/csv")

else:
    st.info("Loading Data...")