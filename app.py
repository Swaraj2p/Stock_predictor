from flask import Flask, render_template, request
from model import predict_stock_price
from datetime import datetime, timedelta
import yfinance as yf

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    error = None
    ticker = None
    currency = None
    date_input = None
    is_historical = False
    arrow = None  # To store the direction of the arrow

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        date_input = request.form['date']

        try:
            date = datetime.strptime(date_input, "%Y-%m-%d")
            today = datetime.today()
            stock_info = yf.Ticker(ticker).info
            currency = stock_info.get('currency', 'USD')

            if date < today:
                end_date = date + timedelta(days=1)
                stock_history = yf.Ticker(ticker).history(start=date_input, end=end_date.strftime("%Y-%m-%d"))

                if not stock_history.empty:
                    predicted_price = stock_history['Close'].values[0]
                    is_historical = True
                else:
                    error = "No historical data found for the specified date."

            else:
                # Fetch the previous day's data for comparison
                prev_date = date - timedelta(days=1)
                prev_stock_history = yf.Ticker(ticker).history(start=prev_date.strftime("%Y-%m-%d"), end=date.strftime("%Y-%m-%d"))

                if not prev_stock_history.empty:
                    prev_price = prev_stock_history['Close'].values[0]
                else:
                    prev_price = None

                predicted_price = predict_stock_price(ticker, date)

                # Determine the arrow direction
                if prev_price is not None:
                    if predicted_price > prev_price:
                        arrow = "up"
                    elif predicted_price < prev_price:
                        arrow = "down"
                    else:
                        arrow = "same"
                else:
                    arrow = "unknown"

        except ValueError:
            error = "Invalid date format. Please use YYYY-MM-DD."
        except Exception as e:
            error = str(e)

    return render_template('index.html', predicted_price=predicted_price, ticker=ticker, currency=currency, date_input=date_input, is_historical=is_historical, error=error, arrow=arrow)

if __name__ == '__main__':
    app.run(debug=True)
