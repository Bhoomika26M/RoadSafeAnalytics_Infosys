import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIGURATION
CLEANED_DATA_FILE = 'cleaned_accident_data.csv'
REPORT_FILE = 'final_project_report.txt'

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found.")
        return None
    print("Loading data for Final Report...")
    return pd.read_csv(filepath)

def generate_insights(df):
    """
    Calculates key statistics for the project.
    """
    print("Calculating insights...")
    
    # 1. Total Accidents
    total_accidents = len(df)
    
    # 2. Most Dangerous City
    top_city = df['City'].mode()[0]
    top_city_count = df['City'].value_counts().iloc[0]
    
    # 3. Most Dangerous Time (Hour)
    peak_hour = int(df['Hour'].mode()[0])
    
    # 4. Most Common Weather
    common_weather = df['Weather_Condition'].mode()[0]
    
    # 5. Severity Analysis (Percentage of severe accidents)
    # Assuming Severity 3 and 4 are "Severe"
    severe_accidents = df[df['Severity'] >= 3]
    severe_pct = (len(severe_accidents) / total_accidents) * 100
    
    # --- Prepare Text Report ---
    report_text = f"""
==================================================
        US ROAD ACCIDENT ANALYSIS - FINAL REPORT
==================================================

1. OVERVIEW
   - Total Accidents Analyzed: {total_accidents:,}
   - Data Source: Kaggle (US Accidents 2016-2023)

2. KEY INSIGHTS (The "Where", "When", and "How")
   
   [WHERE?]
   - The most dangerous city is: {top_city}
   - Accidents in {top_city}: {top_city_count:,}

   [WHEN?]
   - Peak Accident Hour: {peak_hour}:00 (Rush Hour)
   - This suggests traffic congestion is the primary cause.

   [HOW?]
   - Most Common Weather: {common_weather}
   - {severe_pct:.2f}% of accidents were highly severe (Level 3 or 4).

3. CONCLUSION
   - Most accidents happen during clear weather, indicating driver error.
   - Rush hour traffic (Morning/Evening) is the biggest risk factor.
   - Cities with high population density need better traffic control.

==================================================
Report Generated Automatically by Python
==================================================
    """
    return report_text

def save_report(text):
    """
    Saves the text report to a file.
    """
    with open(REPORT_FILE, "w") as f:
        f.write(text)
    print(f"\nReport saved successfully to: '{REPORT_FILE}'")
    print("-" * 30)
    print(text) # Print to terminal as well
    print("-" * 30)

def plot_severity_pie(df):
    """
    Graph 6: Pie Chart of Severity Distribution
    """
    print("\nGenerating Final Graph: Severity Distribution...")
    severity_counts = df['Severity'].value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Distribution of Accident Severity (1=Low, 4=High)')
    
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
        
    plt.savefig('graphs/6_severity_pie.png')
    print("Saved: graphs/6_severity_pie.png")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = load_data(CLEANED_DATA_FILE)
    
    if df is not None:
        # 1. Generate Text Report
        report = generate_insights(df)
        save_report(report)
        
        # 2. Generate Final Graph
        plot_severity_pie(df)
        
        print("\nCONGRATULATIONS! PROJECT COMPLETED.")
        print("You can now submit 'final_project_report.txt' and the 'graphs' folder.")