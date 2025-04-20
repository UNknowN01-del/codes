import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load AirPassengers dataset from R datasets
data = sm.datasets.get_rdataset("AirPassengers").data

# Convert to datetime and set index
data['Month'] = pd.date_range(start='1949-01', periods=len(data), freq='M')
data.set_index('Month', inplace=True)

# Rename column for clarity
data.rename(columns={'value': 'Passengers'}, inplace=True)

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Passengers'], marker='o', linestyle='-', color='blue')
plt.title("AirPassengers Time Series")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate 12-month moving average
data['12_Month_MA'] = data['Passengers'].rolling(window=12).mean()

# Print first few moving average values
print("12-Month Moving Average:\n", data['12_Month_MA'].dropna().head())

# Plot with moving average
plt.figure(figsize=(10, 5))
plt.plot(data['Passengers'], label='Original', color='blue')
plt.plot(data['12_Month_MA'], label='12-Month Moving Average', color='red')
plt.title("AirPassengers with 12-Month Moving Average")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
