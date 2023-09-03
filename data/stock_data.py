import yfinance as yf


def get_stock_data(ticker, period="700d", interval="1h"):
    t = yf.Ticker(ticker)
    data = t.history(period=period, interval=interval)
    return data


def get_today_stock_data(ticker, interval="1h"):
    t = yf.Ticker(ticker)
    data = t.history(period="1d", interval=interval)
    return data

