import pandas as pd

df = pd.read_csv("data/US_Accidents.csv")

# Drop columns with large missing values
df = df.drop(columns=['End_Lat', 'End_Lng', 'Wind_Chill(F)', 'Precipitation(in)'])

# Convert date columns to datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'], errors='coerce')

# Create new features
df['Hour'] = df['Start_Time'].dt.hour
df['Weekday'] = df['Start_Time'].dt.day_name()
df['Month'] = df['Start_Time'].dt.month_name()

# Fill numeric missing values
num_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical missing values
cat_cols = ['City', 'Wind_Direction', 'Weather_Condition', 'Timezone']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates()

print("Remaining missing values:")
print(df.isnull().sum())

print("Data preprocessing completed successfully.")
