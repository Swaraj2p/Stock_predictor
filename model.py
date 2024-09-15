import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# Step 1: Fetch Stock and Market Index Data
def fetch_stock_data(ticker='AAPL', period="5y", index_ticker='^GSPC'):
    stock = yf.Ticker(ticker)
    index = yf.Ticker(index_ticker)
    
    stock_data = stock.history(period=period)
    index_data = index.history(period=period)

    if stock_data.empty or index_data.empty:
        raise Exception(f"No data found for ticker symbol '{ticker}' or index '{index_ticker}'")
    
    # Merging stock and index data based on date
    stock_data['Index'] = index_data['Close']
    
    return stock_data

# Step 2: Prepare Data with More Indicators (MA, RSI, MACD, Trade Volume, Market Index)
def prepare_data(data):
    data['Target'] = data['Close'].shift(-1)  # Next day's close as target
    
    # Moving Averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    
    # Trade Volume
    data['Volume'] = data['Volume']
    
    # Market Index
    data['MarketIndex'] = data['Index']

    # Removing rows with missing data
    data = data.dropna()

    # Feature set with technical indicators
    X = data[['Close', 'MA50', 'MA200', 'RSI', 'MACD', 'Volume', 'MarketIndex']]
    y = data['Target']

    # Scaling the features for LSTM
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Scaling the target for LSTM
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    return X_scaled, y_scaled, scaler_X, scaler_y, data

# Step 3: Build and Train LSTM Model
def train_lstm_model(X_train, y_train):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=55, verbose=1)  # Increased epochs to 50
    
    return model

# Step 4: Predict the Stock Price for a Specific Date (including future dates)
def predict_price_for_date(model, data, scaler_X, scaler_y, date):
    date_str = date.strftime('%Y-%m-%d')

    if date_str not in data.index:
        last_row = data.iloc[-1]

        future_features = {
            'Close': last_row['Close'],
            'MA50': data['Close'].rolling(window=50).mean().iloc[-1],
            'MA200': data['Close'].rolling(window=200).mean().iloc[-1],
            'RSI': last_row['RSI'],
            'MACD': last_row['MACD'],
            'Volume': last_row['Volume'],
            'MarketIndex': last_row['MarketIndex'],
        }
        features = pd.DataFrame([future_features])
        features_scaled = scaler_X.transform(features)
    else:
        features = data.loc[date_str, ['Close', 'MA50', 'MA200', 'RSI', 'MACD', 'Volume', 'MarketIndex']]
        features_scaled = scaler_X.transform([features.values])

    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], features_scaled.shape[1], 1))
    prediction = model.predict(features_scaled)

    # Inverse transform the predicted value to get the actual stock price
    predicted_price = scaler_y.inverse_transform(prediction)
    
    return predicted_price[0][0]

# Step 5: Predict Stock Price for a Specific Ticker and Date
def predict_stock_price(ticker, date, index_ticker='^GSPC'):
    stock_data = fetch_stock_data(ticker, index_ticker=index_ticker)
    X, y, scaler_X, scaler_y, full_data = prepare_data(stock_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = train_lstm_model(X_train, y_train)
    predicted_price = predict_price_for_date(model, full_data, scaler_X, scaler_y, date)
    
    return predicted_price
