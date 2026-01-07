import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIGURATION
# REPLACE with your actual file name downloaded from Kaggle
DATASET_FILE = 'US_Accidents_March23.csv' 

def load_data(filepath):
    """
    Week 1: Dataset Acquisition and Exploration [cite: 29, 31]
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found. Please download it from Kaggle.")
        return None
    
    print("Loading dataset... (This may take a moment due to size)")
    # data types optimization to save memory
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully!")
    return df

def explore_data(df):
    """
    Week 1: Explore structure, shape, and basic stats 
    """
    print("\n--- Data Exploration ---")
    print(f"Shape of dataset: {df.shape}")
    print("\nColumns and Data Types:")
    print(df.info())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nMissing Values Summary (Top 10 columns):")
    missing_values = df.isnull().sum().sort_values(ascending=False)
    print(missing_values.head(10))
    return missing_values

def clean_and_preprocess(df, missing_values):
    """
    Week 2: Data Cleaning and Preprocessing [cite: 34]
    """
    print("\n--- Data Cleaning & Preprocessing ---")
    
    # 1. Drop columns with excessive missing values (>40%) 
    threshold = 0.4 * len(df)
    cols_to_drop = missing_values[missing_values > threshold].index
    print(f"Dropping columns with >40% missing values: {list(cols_to_drop)}")
    df = df.drop(columns=cols_to_drop)
    
    # 2. Drop Duplicate Entries 
    print("Removing duplicate rows...")
    df.drop_duplicates(inplace=True)
    
    # 3. Handle specific missing values (Example: dropping rows where key info like City is missing)
    # You can also impute (fill) values here if preferred
    df.dropna(subset=['City', 'Sunrise_Sunset', 'Weather_Condition'], inplace=True)
    
    # 4. Convert Datetime columns 
    print("Converting timestamp columns...")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    # 5. Create New Features: Hour, Weekday, Month 
    print("Extracting features (Hour, Weekday, Month)...")
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month_name()
    df['Year'] = df['Start_Time'].dt.year
    
    # 6. Basic Outlier Check (Example: Duration) 
    # Calculate duration in minutes
    df['Duration_Minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    
    # Filter out negative durations or absurdly long ones (cleaning logic)
    original_count = len(df)
    df = df[(df['Duration_Minutes'] > 0) & (df['Duration_Minutes'] < 1440)] # Keep < 24 hours
    print(f"Removed {original_count - len(df)} rows with invalid durations.")

    return df

# MAIN EXECUTION
if __name__ == "__main__":
    # Step 1: Load
    df = load_data(DATASET_FILE)
    
    if df is not None:
        # Step 2: Explore (Week 1)
        missing_vals = explore_data(df)
        
        # Step 3: Clean & Preprocess (Week 2)
        df_cleaned = clean_and_preprocess(df, missing_vals)
        
        # Step 4: Verification
        print("\n--- Post-Cleaning Summary ---")
        print(df_cleaned.info())
        print(df_cleaned[['Start_Time', 'Hour', 'Weekday', 'Month']].head())
        
        # Save cleaned data for Milestone 2
        print("\nSaving cleaned data to 'cleaned_accident_data.csv'...")
        df_cleaned.to_csv('cleaned_accident_data.csv', index=False)
        print("Milestone 1 Complete!")
      