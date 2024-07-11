import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_astra_and_competitors.csv')
print(df.head())

# Compute summary statistics
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

# Visualize distributions
plt.figure(figsize=(12, 6))

# Histogram of Price
plt.subplot(2, 2, 1)
plt.hist(df['Price'], bins=10, edgecolor='black')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Boxplot of Milage
plt.subplot(2, 2, 2)
sns.boxplot(x=df['Milage'])
plt.title('Boxplot of Milage')

# Histogram of Power
plt.subplot(2, 2, 3)
plt.hist(df['Power'], bins=5, edgecolor='black')
plt.title('Distribution of Power')
plt.xlabel('Power')
plt.ylabel('Frequency')

# Boxplot of Engine
plt.subplot(2, 2, 4)
sns.boxplot(x=df['Engine'])
plt.title('Boxplot of Engine')

plt.tight_layout()
plt.show()

# Feature Engineering
df['Age'] = 2024 - df['Year']
df['Mileage_per_Year'] = df['Milage'] / df['Age']

print("Updated DataFrame with new features:\n", df.head())

# Scatter plot of Price vs. Milage
plt.figure(figsize=(8, 6))
plt.scatter(df['Milage'], df['Price'], color='blue', alpha=0.5)
plt.title('Price vs. Milage')
plt.xlabel('Milage')
plt.ylabel('Price')
plt.grid(True)
plt.show()