import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Road Accident Analysis", layout="wide")
st.title("🚦 Road Accident Analysis – Milestone 3")

df = pd.read_csv("US_Accidents_March23.csv")

# Fix datetime issue
df['Start_Time_clean'] = df['Start_Time'].astype(str).str.split('.').str[0]
df['datetime'] = pd.to_datetime(df['Start_Time_clean'])
df['Hour'] = df['datetime'].dt.hour

st.header("📍 Week 5: Geospatial Analysis")

st.subheader("Accident Hotspots (Latitude vs Longitude)")
fig1 = plt.figure()
plt.scatter(df['Start_Lng'], df['Start_Lat'], alpha=0.3)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Accident Hotspots")
st.pyplot(fig1)

st.subheader("Top 5 Accident-Prone States")
top_states = df['State'].value_counts().head(5)
fig2 = plt.figure()
top_states.plot(kind='bar')
plt.xlabel("State")
plt.ylabel("Number of Accidents")
plt.title("Top 5 States")
st.pyplot(fig2)


st.header("📊 Week 6: Insights & Hypothesis Testing")

st.subheader("Accidents by Time of Day")
fig3 = plt.figure()
df['Hour'].value_counts().sort_index().plot(kind='line')
plt.xlabel("Hour")
plt.ylabel("Accidents")
plt.title("Accidents vs Time")
st.pyplot(fig3)

st.subheader("Severity during Rain and Fog")
weather_severity = df.groupby('Weather_Condition')['Severity'].mean()

st.write("Average Severity:")
st.write(weather_severity.loc[['Rain', 'Fog']])

st.subheader("Correlation: Visibility vs Severity")
correlation = df[['Visibility(mi)', 'Severity']].corr()
st.write(correlation)


st.success("Milestone 3 Completed: Geospatial Analysis + Insights Successfully Visualized")
