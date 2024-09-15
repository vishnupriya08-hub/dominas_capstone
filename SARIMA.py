import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.metrics import mean_absolute_percentage_error

def perform_eda(file_path):
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

    # Extract features from order_date
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['day'] = df['order_date'].dt.day
    df['weekday'] = df['order_date'].dt.weekday  # 0=Monday, 6=Sunday
    df['quarter'] = df['order_date'].dt.quarter
    df['day_of_year'] = df['order_date'].dt.day_of_year
    df['week_of_year'] = df['order_date'].dt.isocalendar().week
    df['is_weekend'] = df['order_date'].dt.weekday >= 5  # True if weekend, False otherwise

    # Convert integer columns to float
    df[['year', 'month', 'day', 'weekday', 'quarter', 'day_of_year', 'week_of_year']] = df[['year', 'month', 'day', 'weekday', 'quarter', 'day_of_year', 'week_of_year']].astype(float)

    # Interpolate missing values
    df[['year', 'month', 'day', 'weekday', 'quarter', 'day_of_year', 'week_of_year']] = df[['year', 'month', 'day', 'weekday', 'quarter', 'day_of_year', 'week_of_year']].interpolate(method='linear')

    df['is_weekend'] = df['is_weekend'].ffill().bfill()
    df['order_date'] = df['order_date'].ffill().bfill()

    # Convert order_time to datetime
    df['order_time'] = pd.to_datetime(df['order_time'], format='%H:%M:%S', errors='coerce')

    # Extract hour from order_time
    df['order_hour'] = df['order_time'].dt.hour

    # Define time buckets
    def time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    # Apply the function to create a new column
    df['time_of_day'] = df['order_hour'].apply(time_of_day)

    # Extract weekday
    df['order_weekday'] = df['order_date'].dt.day_name()

    # Time series analysis with SARIMA
    df.set_index('order_date', inplace=True)
    monthly_sales = df['total_price'].resample('M').sum()

    # Train-test split
    train_size = int(len(monthly_sales) * 0.8)
    train, test = monthly_sales[:train_size], monthly_sales[train_size:]

    # Check the length of training data
    if len(train) <= 1:
        raise ValueError("Training data is too short for seasonal differencing. Ensure you have sufficient data.")

    # Hyperparameter tuning with auto_arima for SARIMA
    model = pm.auto_arima(train, seasonal=True,
                          trace=True, error_action='ignore', 
                          suppress_warnings=True, stepwise=True, 
                          cv=True)
    print(model.summary())

    # Fit the model
    sarima_model = SARIMAX(train, order=model.order, seasonal_order=model.seasonal_order).fit(disp=False)

    # Forecast the next periods
    forecast = sarima_model.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Plot the forecast
    plt.figure(figsize=(14, 8))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(forecast_mean, label='Forecast')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title('SARIMA Forecast')
    plt.show()

    # Calculate and print the evaluation metrics
    mape = mean_absolute_percentage_error(test, forecast_mean)
    print(f'Mean Absolute Percentage Error: {mape}')

    return df

# Example usage
file_path = 'pizza_sales.csv'
processed_df = perform_eda(file_path)
