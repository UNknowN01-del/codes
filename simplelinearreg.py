import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target
print(df.head())
# Use 1 feature: AveRooms
X = df[['AveRooms']]
y = df['MedHouseVal']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict house value for 5 rooms
prediction = model.predict([[5]])
print("Predicted Median House Value for 5 rooms:", prediction[0])

# Plot data and regression line
plt.scatter(X, y, color='skyblue', label='Actual data', s=10)
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(5, prediction, color='green', label='Prediction (5 rooms)', zorder=5)
plt.xlabel('Average Rooms')
plt.ylabel('Median House Value')
plt.title('Simple Linear Regression: AveRooms vs MedHouseVal')
plt.legend()
plt.grid(True)
plt.show()