import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import product

def prepare_data(file_path):
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

    # Resample to monthly sales
    df.set_index('order_date', inplace=True)
    monthly_sales = df['total_price'].resample('M').sum()

    return monthly_sales

def evaluate_arima(series):
    p = range(0, 3)  # AR term
    d = range(0, 2)  # Differencing term
    q = range(0, 3)  # MA term

    best_params = None
    best_score = float('inf')
    tscv = TimeSeriesSplit(n_splits=5)

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
                continue
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    final_model = ARIMA(series, order=best_params['order'])
    final_model_fit = final_model.fit()
    forecast_steps = 12
    forecast = final_model_fit.forecast(steps=forecast_steps)
    
    y_true = series[-forecast_steps:]
    y_pred = forecast
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return best_params, mape

def evaluate_sarima(series):
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    model = pm.auto_arima(train, seasonal=True, trace=True, error_action='ignore', 
                          suppress_warnings=True, stepwise=True, cv=True)

    sarima_model = SARIMAX(train, order=model.order, seasonal_order=model.seasonal_order).fit(disp=False)

    forecast = sarima_model.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    mape = mean_absolute_percentage_error(test, forecast_mean)
    
    return model.summary(), mape

def evaluate_xgboost(file_path):
    df = prepare_data(file_path)
    monthly_sales = df.reset_index()
    monthly_sales['month'] = monthly_sales['order_date'].dt.month
    monthly_sales['year'] = monthly_sales['order_date'].dt.year

    X = monthly_sales[['month', 'year']]
    y = monthly_sales['total_price']

    tscv = TimeSeriesSplit(n_splits=5)
    model = XGBRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)

    mape = mean_absolute_percentage_error(y, y_pred)
    
    return best_params, mape

def evaluate_prophet(file_path):
    df = prepare_data(file_path)
    prophet_df = df.reset_index().rename(columns={'order_date': 'ds', 'total_price': 'y'})

    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    best_params = None
    best_score = float('inf')
    tscv = TimeSeriesSplit(n_splits=5)

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

    final_model = Prophet(**best_params)
    final_model.fit(prophet_df)
    future = final_model.make_future_dataframe(periods=12, freq='M')
    forecast = final_model.predict(future)
    
    y_true = prophet_df['y']
    y_pred = forecast['yhat'][:len(prophet_df)]
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return best_params, mape

def compare_models(file_path):
    print("Evaluating ARIMA...")
    arima_params, arima_mape = evaluate_arima(prepare_data(file_path))
    print("ARIMA Best Parameters:", arima_params)
    print("ARIMA MAPE:", arima_mape)

    print("\nEvaluating SARIMA...")
    sarima_params, sarima_mape = evaluate_sarima(prepare_data(file_path))
    print("SARIMA Summary:", sarima_params)
    print("SARIMA MAPE:", sarima_mape)

    print("\nEvaluating XGBoost...")
    xgboost_params, xgboost_mape = evaluate_xgboost(file_path)
    print("XGBoost Best Parameters:", xgboost_params)
    print("XGBoost MAPE:", xgboost_mape)

    print("\nEvaluating Prophet...")
    prophet_params, prophet_mape = evaluate_prophet(file_path)
    print("Prophet Best Parameters:", prophet_params)
    print("Prophet MAPE:", prophet_mape)

# Example usage
file_path = 'pizza_sales.csv'
compare_models(file_path)
