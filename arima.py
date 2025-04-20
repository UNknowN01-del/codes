import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
data = sm.datasets.get_rdataset("AirPassengers").data
data['Month'] = pd.date_range(start='1949-01', periods=len(data), freq='M')
data.set_index('Month', inplace=True)
data.rename(columns={'value': 'Passengers'}, inplace=True)

# Time series
series = data['Passengers']

# Fit ARIMA model
model = ARIMA(series, order=(2, 1, 2))
model_fit = model.fit()

# Forecast next 12 months
forecast = model_fit.forecast(steps=12)
print("Next 12 months forecast:\n", forecast)

# Future date index
future_dates = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='M')

# Plot
plt.plot(series)
plt.plot(future_dates, forecast, color='red')
plt.title("ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.show()

