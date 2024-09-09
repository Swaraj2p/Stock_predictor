import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Step 1: Fetch Stock Data
def fetch_stock_data(ticker='AAPL', period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    if data.empty:
        raise Exception(f"No data found for ticker symbol '{ticker}'")
    return data

# Step 2: Prepare Data with Moving Averages as Features
def prepare_data(data):
    data['Target'] = data['Close'].shift(-1)  # Next day's close as target
    data['MA50'] = data['Close'].rolling(window=50).mean()  # 50-day moving average
    data['MA200'] = data['Close'].rolling(window=200).mean()  # 200-day moving average
    data = data.dropna()  # Remove rows with missing data

    X = data[['Close', 'MA50', 'MA200']]  # Features
    y = data['Target']  # Target

    return X, y, data

# Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Predict the Stock Price for a Specific Date (including future dates)
def predict_price_for_date(model, data, date):
    date_str = date.strftime('%Y-%m-%d')

    if date_str not in data.index:
        last_row = data.iloc[-1]  # Get the latest available data

        # Predicting future features (assuming future Close price is same as last known price)
        future_features = {
            'Close': last_row['Close'],
            'MA50': data['Close'].rolling(window=50).mean().iloc[-1],
            'MA200': data['Close'].rolling(window=200).mean().iloc[-1],
        }
        features = pd.DataFrame([future_features]).values
    else:
        features = data.loc[date_str, ['Close', 'MA50', 'MA200']].values.reshape(1, -1)

    prediction = model.predict(features)
    return prediction[0]

def predict_stock_price(ticker, date):
    stock_data = fetch_stock_data(ticker)  # Fetch data for the specified ticker
    X, y, full_data = prepare_data(stock_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_model(X_train, y_train)

    predicted_price = predict_price_for_date(model, full_data, date)
    return predicted_price
