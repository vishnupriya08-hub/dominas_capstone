import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from prophet.diagnostics import cross_validation, performance_metrics

def perform_eda_with_prophet(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Data Cleaning
    df = df.drop(['pizza_id', 'order_id', 'unit_price'], axis=1, errors='ignore')
    df['pizza_name_id'] = df['pizza_name_id'].ffill().bfill()
    df['pizza_category'] = df['pizza_category'].ffill().bfill()
    df['pizza_ingredients'] = df['pizza_ingredients'].ffill().bfill()
    df['pizza_name'] = df['pizza_name'].ffill().bfill()
    df['total_price'] = df['total_price'].interpolate(method='linear')

    # Convert order_date to datetime
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # Prepare data for Prophet
    df.set_index('order_date', inplace=True)
    monthly_sales = df['total_price'].resample('M').sum().reset_index()
    prophet_df = monthly_sales.rename(columns={'order_date': 'ds', 'total_price': 'y'})

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    best_params = None
    best_score = float('inf')
    tscv = TimeSeriesSplit(n_splits=5)

    # Grid search with cross-validation
    for params in ParameterGrid(param_grid):
        scores = []
        for train_index, val_index in tscv.split(prophet_df):
            train_data, val_data = prophet_df.iloc[train_index], prophet_df.iloc[val_index]
            model = Prophet(**params)
            model.fit(train_data)
            forecast = model.predict(val_data)
            y_true = val_data['y']
            y_pred = forecast['yhat']
            mse = mean_squared_error(y_true, y_pred)
            scores.append(mse)
        avg_score = sum(scores) / len(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    # Ensure that best_params is defined before proceeding
    if best_params is None:
        raise ValueError("No best parameters found during grid search.")

    # Train the final model with the best parameters
    final_model = Prophet(**best_params)
    final_model.fit(prophet_df)

    # Forecasting
    future = final_model.make_future_dataframe(periods=12, freq='M')
    forecast = final_model.predict(future)

    # Plot the forecast
    fig = final_model.plot(forecast)
    plt.title('Sales Forecast with Prophet')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.show()

    # Print the best parameters and MAPE
    y_true = prophet_df['y']
    y_pred = forecast['yhat'][:len(prophet_df)]
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("Best parameters:", best_params)
    print("MAPE:", mape)

    # Check if there's enough data for cross-validation
    if len(prophet_df) >= 365:  # Adjust horizon as needed
        # Model diagnostics
        cv_results = cross_validation(final_model, initial='365 days', period='180 days', horizon='90 days')
        performance = performance_metrics(cv_results)
        print("Model performance metrics:")
        print(performance)
    else:
        print("Not enough data for cross-validation.")

# Example usage
file_path = 'pizza_sales.csv'
perform_eda_with_prophet(file_path)
