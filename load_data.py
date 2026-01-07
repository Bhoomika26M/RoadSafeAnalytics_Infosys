import pandas as pd

# Load CSV
df = pd.read_csv("data/US_Accidents.csv")

# Check first rows
print(df.head())

# Check columns
print(df.columns)

# Basic info
print(df.info())
