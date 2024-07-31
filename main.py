# main.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data.dropna(inplace=True)
    return data

def plot_time_series(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Closing Price')
    plt.title(f'{ticker} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def calculate_volatility(data):
    data['Daily Return'] = data['Close'].pct_change()
    plt.figure(figsize=(12, 6))
    plt.plot(data['Daily Return'], label='Daily Return')
    plt.title('Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.show()

    data['Volatility'] = data['Daily Return'].rolling(window=252).std() * np.sqrt(252)
    plt.figure(figsize=(12, 6))
    plt.plot(data['Volatility'], label='Volatility')
    plt.title('Historical Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

def analyze_seasonality(data):
    data['Month'] = data.index.month
    data['Year'] = data.index.year

    monthly_seasonality = data.groupby('Month')['Close'].mean()
    plt.figure(figsize=(12, 6))
    monthly_seasonality.plot(kind='bar')
    plt.title('Monthly Seasonality')
    plt.xlabel('Month')
    plt.ylabel('Average Closing Price')
    plt.show()

def predict_prices(data):
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

    X = data[['Date']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Actual Prices')
    plt.plot(data.index[len(X_train):len(X_train)+len(y_test)], y_pred, label='Predicted Prices', linestyle='--')
    plt.title('Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    data = download_data(ticker, start_date, end_date)
    data = preprocess_data(data)

    plot_time_series(data, ticker)
    calculate_volatility(data)
    analyze_seasonality(data)
    predict_prices(data)

if __name__ == '__main__':
    main()
