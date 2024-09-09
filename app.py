from flask import Flask, render_template, request
from model import predict_stock_price
from datetime import datetime

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    error = None
    ticker = None

    if request.method == 'POST':
        # Get the ticker and date from the form
        ticker = request.form['ticker'].upper()
        date_input = request.form['date']

        # Validate date format
        try:
            date = datetime.strptime(date_input, "%Y-%m-%d")
            predicted_price = predict_stock_price(ticker, date)  # Call model with ticker and date
        except ValueError:
            error = "Invalid date format. Please use YYYY-MM-DD."
        except Exception as e:
            error = str(e)  # Catch errors such as invalid tickers or no data

    return render_template('index.html', predicted_price=predicted_price, ticker=ticker, error=error)

if __name__ == '__main__':
    app.run(debug=True)
