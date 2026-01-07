import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIGURATION
# We use the CLEANED data from Milestone 1
CLEANED_DATA_FILE = 'cleaned_accident_data.csv'

# Create a folder to save graphs if it doesn't exist
if not os.path.exists('graphs'):
    os.makedirs('graphs')

def load_cleaned_data(filepath):
    """
    Load the cleaned dataset efficiently.
    """
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found. Please run Milestone 1 code first.")
        return None
        
    print("Loading cleaned dataset... (This might take 1-2 minutes)")
    # Optimize memory by specifying types if needed, but standard load is usually fine for cleaned data
    df = pd.read_csv(filepath)
    print(f"Data loaded! Shape: {df.shape}")
    return df

def analyze_top_cities(df):
    """
    Graph 1: Top 10 Cities with highest number of accidents
    """
    print("\nGenerating Graph 1: Top 10 Cities...")
    city_counts = df['City'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=city_counts.values, y=city_counts.index, palette='viridis')
    plt.title('Top 10 US Cities by Number of Accidents', fontsize=16)
    plt.xlabel('Number of Accidents', fontsize=12)
    plt.ylabel('City', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('graphs/1_top_cities.png')
    print("Saved: graphs/1_top_cities.png")
    # plt.show() # Uncomment if you want to see the window popup

def analyze_time_trends(df):
    """
    Graph 2: Accidents by Hour of the Day (Identifying Rush Hours)
    """
    print("\nGenerating Graph 2: Accidents by Time of Day...")
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Hour'], bins=24, kde=True, color='orange')
    plt.title('Distribution of Accidents by Hour of Day', fontsize=16)
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.xticks(range(0, 25))
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('graphs/2_accident_hour_trend.png')
    print("Saved: graphs/2_accident_hour_trend.png")

def analyze_weather_conditions(df):
    """
    Graph 3: Top 10 Weather Conditions during accidents
    """
    print("\nGenerating Graph 3: Weather Conditions...")
    weather_counts = df['Weather_Condition'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=weather_counts.values, y=weather_counts.index, palette='coolwarm')
    plt.title('Top 10 Weather Conditions During Accidents', fontsize=16)
    plt.xlabel('Number of Accidents', fontsize=12)
    plt.ylabel('Weather Condition', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('graphs/3_weather_impact.png')
    print("Saved: graphs/3_weather_impact.png")

# MAIN EXECUTION
if __name__ == "__main__":
    # 1. Load Data
    df = load_cleaned_data(CLEANED_DATA_FILE)
    
    if df is not None:
        # Set the visual style
        sns.set_style("whitegrid")
        
        # 2. Run Analysis
        analyze_top_cities(df)
        analyze_time_trends(df)
        analyze_weather_conditions(df)
        