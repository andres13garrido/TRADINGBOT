"""
features.py
Load raw CSV, ensure numeric columns, compute features (SMA, RSI, ATR, lagged returns).
Saves cleaned CSV to data/cleaned/<TICKER>_<INTERVAL>_features.csv
"""
import pandas as pd
from pathlib import Path
from config import RAW_DIR, CLEAN_DIR, TICKER, INTERVAL, SMA_SHORT, SMA_LONG, RSI_PERIOD
from utils import ensure_dirs, setup_logging
import ta
import argparse

ensure_dirs(CLEAN_DIR)
logger = setup_logging(CLEAN_DIR, "features")

def load_csv(path: Path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Drop non-numeric columns (sometimes yfinance includes columns like 'Symbol')
    df = df.select_dtypes(include=["number"])
    df.columns = df.columns.str.lower()  # Standardize to lowercase
    df = df.dropna()
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure required columns exist (now lowercase)
    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    logger.info("Computing technical indicators...")
    df["sma_short"] = df["close"].rolling(window=SMA_SHORT).mean()
    df["sma_long"] = df["close"].rolling(window=SMA_LONG).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=RSI_PERIOD).rsi()
    # ATR via ta
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    # returns and lags
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    for lag in range(1,4):
        df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)
    df = df.dropna()
    logger.info(f"Features computed, {len(df)} rows after dropna")
    return df

def main(ticker=TICKER, interval=INTERVAL):
    raw_path = RAW_DIR / f"{ticker}_{interval}.csv"
    if not raw_path.exists():
        logger.error(f"Raw CSV not found: {raw_path}")
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    df = load_csv(raw_path)
    df_feat = compute_features(df)
    out = CLEAN_DIR / f"{ticker}_{interval}_features.csv"
    df_feat.to_csv(out)
    logger.info(f"Saved features to {out}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument("--interval", default=INTERVAL)
    args = parser.parse_args()
    main(args.ticker, args.interval)