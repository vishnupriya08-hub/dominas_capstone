import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def perform_monthly_sales_prediction(file_path):
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

    # Interpolate missing values for 'year' and 'month'
    df[['year', 'month']] = df[['year', 'month']].interpolate(method='linear')

    # Time series analysis with XGBoost
    df.set_index('order_date', inplace=True)
    monthly_sales = df.resample('M')['total_price'].sum().reset_index()
    monthly_sales['month'] = monthly_sales['order_date'].dt.month
    monthly_sales['year'] = monthly_sales['order_date'].dt.year

    # Features and target
    X = monthly_sales[['month', 'year']]
    y = monthly_sales['total_price']

    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # XGBoost model
    model = XGBRegressor()

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Model training
    best_model.fit(X, y)

    # Predictions
    y_pred = best_model.predict(X)

    # Metrics
    mape = mean_absolute_percentage_error(y, y_pred)

    print(f'Best Parameters: {best_params}')
    print(f'Mean Absolute Percentage Error: {mape}')

    # Plot actual vs predicted
    plt.figure(figsize=(14, 8))
    plt.plot(monthly_sales['order_date'], y, label='Actual')
    plt.plot(monthly_sales['order_date'], y_pred, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.title('Actual vs Predicted Sales')
    plt.show()

# Usage
perform_monthly_sales_prediction('pizza_sales.csv')
