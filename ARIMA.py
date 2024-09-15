#ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

def perform_eda_with_arima(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Data Cleaning
    df = df.drop(['pizza_id', 'order_id', 'unit_price'], axis=1)
    df['pizza_name_id'] = df['pizza_name_id'].ffill().bfill()
    df['pizza_category'] = df['pizza_category'].ffill().bfill()
    df['pizza_ingredients'] = df['pizza_ingredients'].ffill().bfill()
    df['pizza_name'] = df['pizza_name'].ffill().bfill()
    df['total_price'] = df['total_price'].interpolate(method='linear')

    # Convert order_date to datetime
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Time series plot of total sales over time
    plt.figure(figsize=(14, 8))
    df.set_index('order_date')['total_price'].plot()
    plt.title('Total Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.show()

    # Resample to monthly sales
    monthly_sales = df.resample('M', on='order_date')['total_price'].sum()

    # Prepare data for ARIMA
    series = monthly_sales

    # Define parameter grid for ARIMA
    p = range(0, 3)  # AR term
    d = range(0, 2)  # Differencing term
    q = range(0, 3)  # MA term

    best_params = None
    best_score = float('inf')
    tscv = TimeSeriesSplit(n_splits=5)

    # Grid search with cross-validation
    for param in product(p, d, q):
        param = list(param)
        params = {'order': (param[0], param[1], param[2])}
        scores = []
        for train_index, val_index in tscv.split(series):
            train_data, val_data = series.iloc[train_index], series.iloc[val_index]
            try:
                model = ARIMA(train_data, **params)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(val_data))
                mse = mean_squared_error(val_data, forecast)
                scores.append(mse)
            except Exception as e:
                print(f"Error for params {params}: {e}")
                continue
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    # Train the final model with the best parameters
    final_model = ARIMA(series, order=best_params['order'])
    final_model_fit = final_model.fit()

    # Forecasting
    forecast_steps = 12
    forecast = final_model_fit.forecast(steps=forecast_steps)

    # Plot the forecast
    plt.figure(figsize=(14, 8))
    plt.plot(series, label='Historical')
    plt.plot(pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M'), forecast, label='Forecast')
    plt.title('Sales Forecast with ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.show()

    # Print the best parameters and MAPE
    y_true = series[-forecast_steps:]
    y_pred = forecast
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("Best parameters:", best_params)
    print("ARIMA MAPE:", mape)

# Example usage
file_path = 'pizza_sales.csv'
perform_eda_with_arima(file_path)
