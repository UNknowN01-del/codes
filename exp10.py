# 📦 Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Load inbuilt dataset
df = sns.load_dataset("tips")

# 🎯 Set a simple style
sns.set(style="whitegrid")

# 1️⃣ Histogram - distribution of total bill
plt.figure(figsize=(5, 3))
plt.hist(df['total_bill'], bins=10, color='skyblue', edgecolor='black')
plt.title("Histogram of Total Bill")
plt.xlabel("Total Bill")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2️⃣ Bar Chart - average tip by day
plt.figure(figsize=(5, 3))
avg_tip_by_day = df.groupby('day')['tip'].mean().reset_index()
plt.bar(avg_tip_by_day['day'], avg_tip_by_day['tip'], color='coral')
plt.title("Average Tip by Day")
plt.xlabel("Day")
plt.ylabel("Average Tip")
plt.tight_layout()
plt.show()

# 3️⃣ Pie Chart - distribution of smokers
plt.figure(figsize=(4, 4))
smoker_counts = df['smoker'].value_counts()
plt.pie(smoker_counts, labels=smoker_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'orange'])
plt.title("Smoker Distribution")
plt.tight_layout()
plt.show()

# 4️⃣ Box Plot - tip amount by gender
plt.figure(figsize=(5, 3))
sns.boxplot(x='sex', y='tip', data=df, palette='Set2')
plt.title("Box Plot of Tip by Gender")
plt.tight_layout()
plt.show()

# 5️⃣ Violin Plot - total bill by time
plt.figure(figsize=(5, 3))
sns.violinplot(x='time', y='total_bill', data=df, palette='pastel')
plt.title("Violin Plot of Total Bill by Time")
plt.tight_layout()
plt.show()

# 6️⃣ Regression Plot - tip vs total bill
plt.figure(figsize=(5, 3))
sns.regplot(x='total_bill', y='tip', data=df, color='purple')
plt.title("Regression Plot: Tip vs Total Bill")
plt.tight_layout()
plt.show()
