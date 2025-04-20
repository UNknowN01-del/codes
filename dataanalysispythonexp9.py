# 📦 Import libraries
import pandas as pd
import seaborn as sns
import numpy as np

# 📥 Load built-in dataset
df = sns.load_dataset("tips")  # Dataset about restaurant bills and tips

# 👀 Show first few rows
print("First 5 rows:\n")
print(df.head())

# 🧾 Data info
print("\nData Info:\n")
print(df.info())

# 📊 Descriptive statistics
print("\nDescriptive Statistics:\n")
print(df.describe())

# 💡 Add a new column: Tip percentage
df['tip_percent'] = (df['tip'] / df['total_bill']) * 100
print("\nData with Tip Percentage:\n")
print(df[['total_bill', 'tip', 'tip_percent']].head())

# 🧮 NumPy: Mean tip percentage
mean_tip_percent = np.mean(df['tip_percent'])
print("\nAverage Tip Percentage: {:.2f}%".format(mean_tip_percent))

# 🔢 Group by day to see tip averages
print("\nAverage Tip by Day:\n")
print(df.groupby('day')['tip'].mean())
