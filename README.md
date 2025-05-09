# Ex.No: 07 AUTO REGRESSIVE MODEL
## Date:15/04/25
### AIM:
To Implementat an Auto Regressive Model using Python

### ALGORITHM:
Import necessary libraries
Read the CSV file into a DataFrame
Perform Augmented Dickey-Fuller test
Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
Make predictions using the AR model.Compare the predictions with the test data
Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
### NAME : ARCHANA T
### REGISTER NUMBER : 212223240013
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

data = pd.read_csv('powerconsumption.csv', parse_dates=['Datetime'], index_col='Datetime')
ts_column = 'PowerConsumption_Zone1'
ts_data = data[[ts_column]].dropna().iloc[:100]


result = adfuller(ts_data[ts_column])
print('ADF Statistic:', result[0])
print('p-value:', result[1])


x = int(0.8 * len(ts_data))
train_data = ts_data.iloc[:x]
test_data = ts_data.iloc[x:]

lag_order = 13  
model = AutoReg(train_data[ts_column], lags=lag_order)
model_fit = model.fit()

# ACF and PACF plots
plt.figure(figsize=(10, 6))
plot_acf(ts_data[ts_column], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(ts_data[ts_column], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
mse = mean_squared_error(test_data[ts_column], predictions)
print('Mean Squared Error (MSE):', mse)


plt.figure(figsize=(12, 6))
plt.plot(test_data[ts_column], label='Test Data')
plt.plot(predictions, label='Predictions', linestyle='--')
plt.xlabel('Datetime')
plt.ylabel('Power Consumption Zone 1')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/06a0feb3-9df8-4b84-879d-6a56a3b430db)
![image](https://github.com/user-attachments/assets/712e4b12-f6fc-484a-be12-1ef4785a4c35)
![image](https://github.com/user-attachments/assets/684b83d4-8b8f-4b24-a764-799cd1314f39)
![image](https://github.com/user-attachments/assets/dcf44d64-9321-42f5-8f33-041d2aca1a67)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
