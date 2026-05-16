import yfinance as yf
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def download_ticker(ticker, start="2018-01-01", end=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}.parquet")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    df.to_parquet(filepath)
    print(f"Downloaded {len(df)} rows for {ticker}. Saved to {filepath}.")
    return df


def load_ticker(ticker):
    filepath = os.path.join(DATA_DIR, f"{ticker.replace('-', '_')}.parquet")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No data for {ticker}. Run download_ticker() first.")
    return pd.read_parquet(filepath)


if __name__ == "__main__":
    for t in ["AAPL", "BTC-USD", "ETH-USD"]:
        download_ticker(t)