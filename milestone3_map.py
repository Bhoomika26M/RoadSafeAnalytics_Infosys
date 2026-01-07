import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIGURATION
CLEANED_DATA_FILE = 'cleaned_accident_data.csv'

# Create graphs folder if it doesn't exist
if not os.path.exists('graphs'):
    os.makedirs('graphs')

def load_data(filepath):
    """
    Load the cleaned data=
    """
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found. Please run Milestone 1 first.")
        return None
    
    print("Loading dataset for Map Analysis... (Please wait)")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully! Rows: {len(df)}")
    return df

def visualize_usa_map(df):
    """
    Graph 4: Scatter Plot of Latitude/Longitude (Creates a Map of USA)
    NOTE: We take a sample of data to make plotting faster.
    """
    print("\nGenerating Graph 4: Accident Map of USA...")
    
    # Taking a random sample of 100,000 points to prevent computer lagging
    # If your computer is fast, you can increase this number
    sample_df = df.sample(n=100000, random_state=42)
    
    plt.figure(figsize=(12, 8))
    
    # Plotting Longitude on X-axis and Latitude on Y-axis
    sns.scatterplot(
        x='Start_Lng', 
        y='Start_Lat', 
        data=sample_df, 
        hue='Severity', 
        palette='coolwarm', 
        s=10,        # Size of dots
        alpha=0.3    # Transparency
    )
    
    plt.title('Geographic Distribution of Accidents (Sample of 100k)', fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.legend(title='Severity', loc='lower right')
    
    # Save map
    plt.tight_layout()
    plt.savefig('graphs/4_usa_accident_map.png')
    print("Saved: graphs/4_usa_accident_map.png")

def visualize_correlation(df):
    """
    Graph 5: Correlation Heatmap (Relationship between variables)
    """
    print("\nGenerating Graph 5: Correlation Heatmap...")
    
    # Select only numerical columns for correlation
    cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap: Weather vs Severity', fontsize=16)
    
    # Save heatmap
    plt.tight_layout()
    plt.savefig('graphs/5_correlation_heatmap.png')
    print("Saved: graphs/5_correlation_heatmap.png")

# MAIN EXECUTION
if __name__ == "__main__":
    # 1. Load Data
    df = load_data(CLEANED_DATA_FILE)
    
    if df is not None:
        sns.set_style("darkgrid")
        
        # 2. Run Map Visualization
        visualize_usa_map(df)
        
        # 3. Run Correlation Analysis
        visualize_correlation(df)
        
        