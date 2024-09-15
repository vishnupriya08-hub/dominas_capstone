import pandas as pd
from prophet import Prophet

def perform_eda_with_prophet(file_path, ingredient_file_path, purchase_order_file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    ingredients_df = pd.read_csv(ingredient_file_path)

    # Interpolate Items_Qty_In_Grams in ingredients_df
    ingredients_df['Items_Qty_In_Grams'] = ingredients_df['Items_Qty_In_Grams'].interpolate(method='linear')

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
    monthly_sales = df.resample('M', on='order_date')['total_price'].sum().reset_index()
    monthly_sales.columns = ['ds', 'y']  # Rename columns for Prophet

    # Train the Prophet model
    model = Prophet()
    model.fit(monthly_sales)

    # Create a DataFrame for future dates
    future = model.make_future_dataframe(periods=7)  # Forecast for 7 days
    forecast = model.predict(future)

    # Calculate total sales for the forecast period (one week)
    forecast_period = 7  # Forecast period in days
    forecast_df = forecast[['ds', 'yhat']].tail(forecast_period)
    total_sales_one_week = forecast_df['yhat'].sum()

    # Initialize DataFrame for ingredient totals
    ingredient_totals = pd.DataFrame()

    # For each pizza type, calculate ingredient requirements
    for pizza_id in df['pizza_name_id'].unique():
        pizza_data = df[df['pizza_name_id'] == pizza_id]
        
        # Get total forecasted sales for this pizza type
        pizza_sales = pizza_data['total_price'].sum()
        pizza_forecasted_sales = total_sales_one_week * (pizza_sales / df['total_price'].sum())

        # Get ingredient data for this pizza type
        ingredient_data = ingredients_df[ingredients_df['pizza_name_id'] == pizza_id]
        if not ingredient_data.empty:
            ingredient_data = ingredient_data.copy()
            ingredient_data['total_quantity'] = ingredient_data['Items_Qty_In_Grams'] * pizza_forecasted_sales
            ingredient_totals = pd.concat([ingredient_totals, ingredient_data])

    # Aggregate ingredient totals
    purchase_order = ingredient_totals.groupby('pizza_ingredients').agg({'total_quantity': 'sum'}).reset_index()
    purchase_order = purchase_order.sort_values(by='total_quantity', ascending=False)

    # Print the purchase order
    print("Purchase Order:")
    for _, row in purchase_order.iterrows():
        print(f"{row['pizza_ingredients']}: {row['total_quantity']:.2f} grams")

    # Save the purchase order to a CSV file
    purchase_order.to_csv(purchase_order_file_path, index=False)

# Example usage
file_path = 'pizza_sales.csv'
ingredient_file_path = 'pizza_ingredients.csv'
purchase_order_file_path = 'purchase_order.csv'

perform_eda_with_prophet(file_path, ingredient_file_path, purchase_order_file_path)
