"""
fetch_data.py
Fetch historical data using yfinance.
Saves CSV into data/raw/<TICKER>_<INTERVAL>.csv
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from config import RAW_DIR, TICKER, START, END, INTERVAL
from utils import ensure_dirs, setup_logging
import argparse

ensure_dirs(RAW_DIR)
logger = setup_logging(RAW_DIR, "fetch_data")

def fetch_yfinance(ticker=TICKER, start=START, end=END, interval=INTERVAL):
    out = RAW_DIR / f"{ticker}_{interval}.csv"
    logger.info(f"Fetching {ticker} interval={interval} start={start} end={end}")
    
    try:
        # yfinance has intraday limits; daily data is full history
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df is None or df.empty:
            logger.error("No data fetched. Try changing interval or date range.")
            raise RuntimeError("No data fetched.")
        
        # Handle multi-index columns (yfinance sometimes returns these)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            logger.info("Flattened multi-index columns")
        
        # Save
        df.to_csv(out)
        logger.info(f"Saved raw CSV to {out} with {len(df)} rows")
        return out
    except Exception as e:
        logger.exception(f"Error fetching data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--interval", default=INTERVAL)
    args = parser.parse_args()
    fetch_yfinance(args.ticker, args.start, args.end, args.interval)