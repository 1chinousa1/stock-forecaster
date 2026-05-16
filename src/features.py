import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna

FORWARD_DAYS = 5

def add_target(df: pd.DataFrame, forward_days: int = FORWARD_DAYS) -> pd.DataFrame:
    df = df.copy()
    df["future_close"] = df["Close"].shift(-forward_days)
    df["target"] = (df["future_close"] > df["Close"]).astype(int)
    df.drop(columns=["future_close"], inplace=True)
    return df

def build_features(df: pd.DataFrame, forward_days: int = FORWARD_DAYS) -> pd.DataFrame:
    df = df.copy()
    df = dropna(df)

    df = add_all_ta_features(
        df, open="Open", high="High", low="Low",
        close="Close", volume="Volume", fillna=True
    )

    for lag in [1, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df["Close"].pct_change(lag)

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    df = add_target(df, forward_days)
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_ticker

    df = load_ticker("BTC-USD")
    featured = build_features(df)

    print(f"Shape: {featured.shape}")
    print(f"Columns: {len(featured.columns)} features")
    print(f"Target distribution:\n{featured['target'].value_counts(normalize=True).round(3)}")