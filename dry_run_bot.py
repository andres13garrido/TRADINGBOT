"""
dry_run_bot.py
A safe demo loop that uses the trained model to predict on the latest data pulled from yfinance,
and logs the 'would-be' decisions. No orders are sent.
"""
import joblib
import yfinance as yf
import ta
import pandas as pd
import time
from config import TICKER, INTERVAL, MODEL_FILE, ML_THRESHOLD
from utils import setup_logging, ensure_dirs
from pathlib import Path

LOG_DIR = Path("logs")
ensure_dirs(LOG_DIR)
logger = setup_logging(LOG_DIR, "dry_run")

def compute_features_row(df):
    """Compute features on recent data and return last row"""
    df2 = df.copy()
    
    # Handle MultiIndex columns from yfinance
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = df2.columns.get_level_values(0)
    
    # Convert to lowercase
    df2.columns = df2.columns.str.lower()
    
    df2["sma_short"] = df2["close"].rolling(window=10).mean()
    df2["sma_long"] = df2["close"].rolling(window=50).mean()
    df2["rsi"] = ta.momentum.RSIIndicator(df2["close"], window=14).rsi()
    df2["atr"] = ta.volatility.average_true_range(df2["high"], df2["low"], df2["close"], window=14)
    df2["ret_1"] = df2["close"].pct_change(1)
    df2["ret_3"] = df2["close"].pct_change(3)
    
    for lag in range(1,4):
        df2[f"ret_lag_{lag}"] = df2["ret_1"].shift(lag)
    
    df2 = df2.dropna()
    
    if df2.empty:
        raise ValueError("Not enough data to compute features")
    
    return df2.iloc[-1]

def load_model(model_file=MODEL_FILE):
    """Load trained model bundle"""
    bundle = joblib.load(model_file)
    return bundle["model"], bundle["scaler"], bundle["features"]

def fetch_recent(ticker=TICKER, interval=INTERVAL, period="60d"):
    """Fetch recent bars from yfinance"""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("No recent data")
    
    # Check if data is fresh (within last hour for intraday, last day for daily)
    latest_time = df.index[-1]
    now = pd.Timestamp.now(tz=latest_time.tz)
    time_diff = now - latest_time
    
    if interval == "1d":
        max_age = pd.Timedelta(days=5)  # Allow up to 5 days old (weekend)
    else:
        max_age = pd.Timedelta(hours=24)  # For intraday, warn if > 24h old
    
    if time_diff > max_age:
        logger.warning(f"Data is {time_diff} old. Markets may be closed.")
    
    return df

def dry_run_loop(poll_seconds=60):
    """Main loop - fetches data and logs predictions"""
    try:
        model, scaler, features = load_model(MODEL_FILE)
        logger.info("Model loaded successfully")
        logger.info(f"Features: {features}")
        logger.info(f"ML Threshold: {ML_THRESHOLD}")
    except Exception as e:
        logger.exception("Failed to load model")
        return
    
    logger.info("Starting dry-run loop (no orders will be sent).")
    logger.info(f"Polling every {poll_seconds} seconds")
    logger.info(f"Trading {TICKER} with {INTERVAL} interval")
    
    # Check if market hours (for US stocks)
    now = pd.Timestamp.now(tz='America/New_York')
    if now.hour < 9 or now.hour >= 16 or now.weekday() >= 5:
        logger.warning("⚠️  US MARKETS ARE CURRENTLY CLOSED")
        logger.warning(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.warning("Market hours: Mon-Fri 9:30 AM - 4:00 PM EST")
        logger.warning("Price data will be stale until markets open.")
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    prev_price = None
    
    while True:
        try:
            df = fetch_recent()
            last = compute_features_row(df)
            
            X = last[features].values.reshape(1, -1)
            # Convert to DataFrame to preserve feature names (silences sklearn warning)
            X_df = pd.DataFrame(X, columns=features)
            Xs = scaler.transform(X_df)
            prob = model.predict_proba(Xs)[0,1]
            signal = int(prob > ML_THRESHOLD)
            action = "BUY" if signal==1 else "SELL"
            price = float(last["close"])
            
            # Check if price changed
            if prev_price is not None and price == prev_price:
                logger.warning(f"⚠️  Price unchanged ({price:.2f}) - markets likely closed")
            prev_price = price
            
            # Get latest data timestamp
            data_time = df.index[-1]
            
            # Calculate data age - handle timezone properly
            if data_time.tz is not None:
                now = pd.Timestamp.now(tz=data_time.tz)
            else:
                now = pd.Timestamp.now()
            data_age = now - data_time
            
            msg = f"Signal={action} prob={prob:.3f} price={price:.2f} threshold={ML_THRESHOLD} data_age={data_age}"
            logger.info(msg)
            print(msg)
            
            consecutive_errors = 0  # Reset on success
            
        except Exception as e:
            consecutive_errors += 1
            logger.exception(f"Dry-run error ({consecutive_errors}/{max_consecutive_errors})")
            
            if consecutive_errors >= max_consecutive_errors:
                logger.error("Too many consecutive errors, exiting")
                break
        
        time.sleep(poll_seconds)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()
    
    dry_run_loop(poll_seconds=args.interval)