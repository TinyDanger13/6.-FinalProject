import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("final_listings.csv",delimiter=",")

# print(df.head(5))

# print(df["Mileage"])

# Extracting year and converting to number format
def extract_year(date_list):
    date = date_list.replace("[","").replace("]","").replace("'","")
    if '-' in date:
        return int(date.split('-')[0])
    else:
        return int(date)
    
df['Year'] = df["Year"].apply(extract_year)

# print(df["Year"])

# print(df.info())

# print(df["Power_unit"])

# Spliting the 'Power_unit' column into two separate columns and cleaning the data
df[['Engine', 'Power']] = df['Power_unit'].str.split(',\s+', expand=True)

df['Engine'] = df['Engine'].str.extract('(\d+\.\d+)')
df['Power'] = df['Power'].str.extract('(\d+)')

# Converting extracted strings to numeric values. Converting to cc and handling errors (errors turned into Nan values)
df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce') * 1000
df['Power'] = pd.to_numeric(df['Power'], errors='coerce')

# Droping rows with NaN values
df.dropna(inplace=True)

# Converting columns to number format
df['Engine'] = df['Engine'].astype(int)
df['Power'] = df['Power'].astype(int)

# Droping 'Power_unit' column
df.drop(columns=['Power_unit'], inplace=True)

# print(df[['Engine','Power']])

print(df.head())

print(df.info())

# Saving the cleaned data to a CSV file
df.to_csv('final_cleaned.csv', index=False)

print(df.describe())

# Visualizing data
sns.set_theme(style="whitegrid")

# Histogram of engine_size
plt.figure(figsize=(12, 6))
sns.histplot(df['Engine'], bins=20, kde=True)
plt.title('Distribution of Engine Size')
plt.xlabel('Engine Size (cc)')
plt.ylabel('Frequency')
plt.show()

# Histogram of power(kW)
plt.figure(figsize=(12, 6))
sns.histplot(df['Power'], bins=20, kde=True)
plt.title('Distribution of Power')
plt.xlabel('Power (kW)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of engine_size vs power
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['Engine'], y=df['Power'])
plt.title('Engine Size vs Power')
plt.xlabel('Engine Size (cc)')
plt.ylabel('Power (kW)')
plt.show()