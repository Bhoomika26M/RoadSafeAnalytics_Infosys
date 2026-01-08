#Milestone 2 - Week 3
import milestone1 as ms1
import matplotlib.pyplot as plt
import seaborn as sns

#Distribution of Accident Severity
plt.figure(figsize=(8, 5))
sns.countplot(x='Severity', data=ms1.df, palette='viridis', hue='Severity', legend=False)
plt.title('Distribution of Accident Severity')
plt.xlabel('Severity')
plt.ylabel('Number of Accidents')
plt.show()

#Accident Frequency by Day of Week
plt.figure(figsize=(10, 6))
sns.countplot(x='Start_Day_of_Week', data=ms1.df, order=ms1.df['Start_Day_of_Week'].value_counts().index, palette='magma', hue='Start_Day_of_Week', legend=False)
plt.title('Accident Frequency by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

#Accident Frequency by Month
plt.figure(figsize=(10, 6))
sns.countplot(x='Start_Month', data=ms1.df, order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], palette='viridis', hue='Start_Month', legend=False)
plt.title('Accident Frequency by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, ha='right')
plt.show()

#Top 10 Weather Conditions during Accidents
plt.figure(figsize=(12, 6))
top_weather = ms1.df['Weather_Condition'].value_counts().head(10)
sns.barplot(x=top_weather.index, y=top_weather.values, palette='coolwarm', hue=top_weather.index, legend=False)
plt.title('Top 10 Weather Conditions during Accidents')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, ha='right')
plt.show()

#Road types during Accidents
road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

# Calculate the sum of 'True' values for each road feature
road_feature_counts = ms1.df[road_features].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 7))
sns.barplot(x=road_feature_counts.index, y=road_feature_counts.values, palette='coolwarm', hue=road_feature_counts.index, legend=False)
plt.title('Frequency of Road Features at Accident Locations')
plt.xlabel('Road Feature')
plt.ylabel('Number of Accidents (Feature Present)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Geographical Distribution of Accidents (Top 10 Cities)
plt.figure(figsize=(12, 6))
top_cities = ms1.df['City'].value_counts().head(10)
sns.barplot(x=top_cities.index, y=top_cities.values, palette='rocket', hue=top_cities.index, legend=False)
plt.title('Top 10 Cities by Accident Count')
plt.xlabel('City')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, ha='right')
plt.show()

road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

selected_columns = [
    'Severity',
    'Weather_Condition',
    'Visibility(mi)',
    'Start_Hour',
    'Start_Day_of_Week',
    'Sunrise_Sunset'
] + road_features

# Create the new DataFrame df_analysis with only the selected columns
df_analysis = ms1.df[selected_columns].copy()

print(f"Created df_analysis with {df_analysis.shape[1]} columns and {df_analysis.shape[0]} rows.")
print("Columns in df_analysis:", df_analysis.columns.tolist())

#Prepare Data for Weather Conditions Analysis
top_10_weather_conditions = df_analysis['Weather_Condition'].value_counts().head(10).index.tolist()

df_top_weather = df_analysis[df_analysis['Weather_Condition'].isin(top_10_weather_conditions)].copy()

print("Top 10 Weather Conditions:", top_10_weather_conditions)
print(f"\nShape of df_top_weather: {df_top_weather.shape}")
print("\nFirst 5 rows of df_top_weather:")
print(df_top_weather.head())

#Visualize Severity vs. Top Weather Conditions
plt.figure(figsize=(15, 8))
sns.countplot(x='Weather_Condition', hue='Severity', data=df_top_weather, palette='viridis')
plt.title('Accident Severity Distribution Across Top 10 Weather Conditions')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Severity')
plt.tight_layout()
plt.show()

#Visualize Severity vs. Visibility
plt.figure(figsize=(10, 6))
sns.boxplot(x='Severity', y='Visibility(mi)', data=df_analysis, palette='viridis')
plt.title('Accident Severity vs. Visibility (miles)')
plt.xlabel('Severity')
plt.ylabel('Visibility (mi)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Visualize Severity vs. Road Features
sns.set_style("whitegrid")

# Get the top 6 road features based on their counts
top_road_features = road_feature_counts.head(6).index.tolist()

# Determine the number of rows and columns for subplots
num_features = len(top_road_features)
num_cols = 3
num_rows = (num_features + num_cols - 1) // num_cols # Ceiling division

plt.figure(figsize=(num_cols * 5, num_rows * 5))

for i, feature in enumerate(top_road_features):
    plt.subplot(num_rows, num_cols, i + 1)

    # Filter data for when the current road feature is True
    df_feature_present = df_analysis[df_analysis[feature] == True]

    sns.countplot(x='Severity', hue='Severity', data=df_feature_present, palette='viridis', legend=False)
    plt.title(f'Severity Dist. when {feature.replace("_", " ")} is Present')
    plt.xlabel('Severity')
    plt.ylabel('Number of Accidents')

plt.tight_layout()
plt.show()

#Analyze Correlation Between Numerical Variables
correlation_columns = [
    'Severity',
    'Visibility(mi)',
    'Accident_Duration_min',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Wind_Speed(mph)'
]

# Calculate the correlation matrix
correlation_matrix = ms1.df[correlation_columns].corr()

print("Correlation Matrix:")
print(correlation_matrix)

#Visualize Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Numerical Variables and Severity')
plt.tight_layout()
plt.show()

#Visualize Relationships with Pair Plots
pair_plot_cols = ['Severity', 'Visibility(mi)', 'Distance(mi)', 'Temperature(F)']
df_pair_plot = ms1.df[pair_plot_cols].copy()

print(f"Created df_pair_plot with {df_pair_plot.shape[1]} columns and {df_pair_plot.shape[0]} rows.")
print("Columns in df_pair_plot:", df_pair_plot.columns.tolist())

# Generate pair plot
#sns.pairplot(df_pair_plot, hue='Severity', palette='viridis', diag_kind='kde')
#plt.suptitle('Pair Plots of Selected Numerical Variables by Severity', y=1.02) # Adjust suptitle to not overlap
#plt.show()

#Visualize Severity vs. Time of Day (Start Hour)
plt.figure(figsize=(15, 8))
sns.countplot(x='Start_Hour', hue='Severity', data=df_analysis, palette='viridis', order=sorted(df_analysis['Start_Hour'].unique()))
plt.title('Accident Severity Distribution by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.legend(title='Severity')
plt.tight_layout()
plt.show()

