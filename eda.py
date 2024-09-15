import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Plot distribution of total_price
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_price'], kde=True)
    plt.title('Distribution of Total Price')
    plt.show()

    # Plot distribution of quantity
    plt.figure(figsize=(10, 6))
    sns.histplot(df['quantity'], kde=True)
    plt.title('Distribution of Quantity')
    plt.show()

    # Plot categorical variables
    plt.figure(figsize=(10, 6))
    sns.countplot(x='pizza_size', data=df)
    plt.title('Distribution of Pizza Size')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='pizza_category', data=df)
    plt.title('Distribution of Pizza Category')
    plt.show()

    # Scatter plot of total_price vs quantity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='total_price', y='quantity', data=df)
    plt.title('Total Price vs Quantity')
    plt.show()

    # Box plots for categorical variables
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='pizza_size', y='total_price', data=df)
    plt.title('Total Price by Pizza Size')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='pizza_category', y='total_price', data=df)
    plt.title('Total Price by Pizza Category')
    plt.show()

    # Time series plot of total sales over time
    plt.figure(figsize=(14, 8))
    df.set_index('order_date')['total_price'].plot()
    plt.title('Total Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.show()

    # Monthly sales trend
    monthly_sales = df.resample('M', on='order_date')['total_price'].sum()
    plt.figure(figsize=(14, 8))
    monthly_sales.plot()
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.show()

    # Daily sales trend
    daily_sales = df.resample('D', on='order_date')['total_price'].sum()
    plt.figure(figsize=(14, 8))
    daily_sales.plot()
    plt.title('Daily Sales Trend')
    plt.xlabel('Day')
    plt.ylabel('Total Sales')
    plt.show()

    # Drop non-numeric columns
    columns_to_drop = ['order_date', 'order_time', 'pizza_name_id', 'pizza_size', 'pizza_category', 'pizza_ingredients', 'pizza_name']
    numeric_df = df.drop(columns=columns_to_drop)

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Convert order_time to datetime
    df['order_time'] = pd.to_datetime(df['order_time'], format='%H:%M:%S', errors='coerce')

    # Extract hour from order_time
    df['order_hour'] = df['order_time'].dt.hour

    # Plot hourly sales distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(df['order_hour'], bins=24, kde=True)
    plt.title('Hourly Sales Distribution')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Orders')
    plt.show()

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

    # Plot sales by time of day
    plt.figure(figsize=(12, 8))
    sns.countplot(x='time_of_day', data=df, order=['Morning', 'Afternoon', 'Evening', 'Night'])
    plt.title('Sales by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Orders')
    plt.show()

    # Aggregate sales by hour
    hourly_sales = df.groupby(df['order_time'].dt.floor('H'))['quantity'].sum().reset_index()

    # Plot time series of sales
    plt.figure(figsize=(12, 8))
    plt.plot(hourly_sales['order_time'], hourly_sales['quantity'])
    plt.title('Time Series of Sales')
    plt.xlabel('Time')
    plt.ylabel('Quantity Sold')
    plt.xticks(rotation=45)
    plt.show()

    # Extract weekday
    df['order_weekday'] = df['order_date'].dt.day_name()

    # Plot sales by weekday
    plt.figure(figsize=(12, 8))
    sns.countplot(x='order_weekday', data=df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('Sales by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Orders')
    plt.show()

    # Create pivot table for heatmap
    heatmap_data = df.pivot_table(index='order_weekday', columns='order_hour', values='quantity', aggfunc='sum').fillna(0)

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu')
    plt.title('Sales Heatmap by Hour and Day of the Week')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of the Week')
    plt.show()

    # Top 10 pizzas by total quantity sold
    pizza_sales = df.groupby('pizza_name_id')['quantity'].sum().reset_index()
    pizza_sales.columns = ['Pizza Type', 'Total Quantity Sold']
    top_10_pizza_sales = pizza_sales.sort_values(by='Total Quantity Sold', ascending=False).head(10)

    # Plotting the data
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Pizza Type', y='Total Quantity Sold', data=top_10_pizza_sales, palette='viridis')
    plt.title('Top 10 Pizzas by Total Quantity Sold')
    plt.xlabel('Pizza Type')
    plt.ylabel('Total Quantity Sold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Call the function with your CSV file path
perform_eda('pizza_sales.csv')
