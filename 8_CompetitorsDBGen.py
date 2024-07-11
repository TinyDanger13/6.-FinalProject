import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load DFs from files
df_astra = pd.read_csv('more_cleaned_opel_astra.csv')
df_golf = pd.read_csv('vw_golf_cleaned.csv')
df_focus = pd.read_csv('Ford_Focus_cleaned.csv')
df_megane = pd.read_csv('Renault_Megane_cleaned.csv')
df_308 = pd.read_csv('Peugeot_308_cleaned.csv')
df_i30 = pd.read_csv('Hyundai_i30_cleaned.csv')
df_ceed = pd.read_csv('Kia_Ceed_cleaned.csv')
df_3 = pd.read_csv('Mazda_3_cleaned.csv')
df_corolla = pd.read_csv('Toyota_Corolla_cleaned.csv')
df_civic = pd.read_csv('Honda_Civic_cleaned.csv')
df_octavia = pd.read_csv('Skoda_Octavia_cleaned.csv')

# Concatenate DataFrames
merged_df = pd.concat([df_astra, df_golf, df_focus, df_megane, df_308, df_i30, 
                                   df_ceed, df_3, df_corolla, df_civic, df_octavia], 
                                    ignore_index=True)

# Display the merged DataFrame
print(merged_df.info())

# Checking for outliers
rows_to_remove = merged_df[merged_df['Power'] > 500]
# rows_to_remove = merged_df[merged_df['Engine'] < 500]
# rows_to_remove = merged_df[merged_df['Engine'] > 3000]

num_rows_to_remove = rows_to_remove.shape[0]

print(f'Number of rows to be removed: {num_rows_to_remove}')

# Removing outliers
merged_df = merged_df[merged_df['Power'] <= 500]
merged_df = merged_df[merged_df['Engine'] <= 3000]
merged_df = merged_df[merged_df['Engine'] >= 500]

# Saving the cleaned data to a CSV file
merged_df.to_csv('cleaned_astra_and_competitors.csv', index=False)

# Visualizing data
sns.set_theme(style="whitegrid")

# Histogram of engine_size
plt.figure(figsize=(12, 6))
sns.histplot(merged_df['Engine'], bins=20, kde=True)
plt.title('Distribution of Engine Size')
plt.xlabel('Engine Size (cc)')
plt.ylabel('Frequency')
plt.show()

# Histogram of power(kW)
plt.figure(figsize=(12, 6))
sns.histplot(merged_df['Power'], bins=20, kde=True)
plt.title('Distribution of Power')
plt.xlabel('Power (kW)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of engine_size vs power
plt.figure(figsize=(12, 6))
sns.scatterplot(x=merged_df['Engine'], y=merged_df['Power'])
plt.title('Engine Size vs Power')
plt.xlabel('Engine Size (cc)')
plt.ylabel('Power (kW)')
plt.show()