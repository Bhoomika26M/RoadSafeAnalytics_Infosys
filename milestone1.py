# Pull request test
# milestone-1 branch change
import pandas as pd
import numpy as np

df = pd.read_csv("US_Accidents_March23.csv")

print(df.shape)
print( df.columns)

print(df.info())
print(df.describe())
print( df.isna().sum())

print(" CLEANING & PREPROCESSING ")

# 1) Drop columns with >40% missing
missing_percent = df.isna().sum() * 100 / len(df)
cols_drop = missing_percent[missing_percent > 40].index
df = df.drop(columns=cols_drop)
print("Dropped columns:", list(cols_drop))

# 2) Simple fill for remaining missing values
num_cols = df.select_dtypes(include=["number"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

obj_cols = df.select_dtypes(include=["object"]).columns
df[obj_cols] = df[obj_cols].fillna("Unknown")

# 3) Datetime and new features
if "Start_Time" in df.columns:
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["Hour"] = df["Start_Time"].dt.hour
    df["Weekday"] = df["Start_Time"].dt.day_name()
    df["Month"] = df["Start_Time"].dt.month
    print("Created: Hour, Weekday, Month")

# 4) Duplicates
print("Duplicates before:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after:", df.duplicated().sum())

print("\nMilestone 1 (Week 1 + Week 2) steps completed.")
