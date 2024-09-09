# Stock Price Predictor 📈

A web application that predicts future stock prices based on historical data using machine learning. Built using Flask, scikit-learn, and yfinance.

## Features
- 📅 **Date-based Prediction**: Users can input a specific date to get stock price predictions for that date.
- 📊 **Stock Ticker Selection**: Input any stock ticker from Yahoo Finance (e.g., `AAPL`, `GOOGL`) to retrieve predictions.
- ⚙️ **Machine Learning Model**: The app uses a linear regression model trained on historical stock data.
- 🖥️ **Interactive UI**: A user-friendly web interface built with Flask for easy input and results display.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/stock-predictor.git
   cd stock-predictor
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/`.

## Usage

1. Enter the **stock ticker** (e.g., `AAPL` for Apple, `GOOGL` for Alphabet) in the input field.
2. Enter the **date** (in YYYY-MM-DD format) for which you want the stock price prediction.
3. Click **Predict** to get the estimated stock price for that date.

## Project Structure

```
stock_predictor/
├── app.py              # Flask application
├── model.py            # Stock prediction model
├── templates/
│   └── index.html      # HTML template for UI
├── static/             # Static files (CSS, JS, etc.)
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

## Technologies Used

- **Flask**: For building the web interface.
- **scikit-learn**: For the machine learning model (Linear Regression).
- **yfinance**: For fetching historical stock data.
- **pandas**: For data manipulation.

## Future Enhancements

- 🔄 **Additional Machine Learning Models**: Adding support for more complex models like Decision Trees or LSTM.
- 📱 **Mobile-friendly UI**: Improving the design to be more responsive on mobile devices.
- ⏲️ **Live Data Update**: Use real-time stock data for prediction rather than relying solely on historical data.

## Contributing

Feel free to submit issues or pull requests if you'd like to improve the project!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
