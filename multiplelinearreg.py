import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load California Housing data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Select 2 features: AveRooms (average rooms), AveOccup (average occupancy)
X = df[['AveRooms', 'AveOccup']]
y = df['MedHouseVal']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predict for AveRooms=5, AveOccup=3
prediction = model.predict([[5, 3]])
print("Predicted Median House Value for AveRooms=5 and AveOccup=3:", prediction[0])

# Plot 1: House Value vs AveRooms
plt.subplot(1, 2, 1)
plt.scatter(df['AveRooms'], y, color='blue', s=10)
plt.xlabel('Average Rooms')
plt.ylabel('Median House Value')
plt.title('Value vs Rooms')
plt.grid(True)

# Plot 2: House Value vs AveOccup
plt.subplot(1, 2, 2)
plt.scatter(df['AveOccup'], y, color='green', s=10)
plt.xlabel('Average Occupancy')
plt.ylabel('Median House Value')
plt.title('Value vs Occupancy')
plt.grid(True)

plt.tight_layout()
plt.show()