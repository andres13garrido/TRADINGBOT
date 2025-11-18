"""
ibkr_bot.py
Connects to Interactive Brokers via ib_insync, uses the trained model to make decisions,
and places orders if dry_run=False. ALWAYS test with dry_run=True first.
"""
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import joblib
import time
from config import (IB_HOST, IB_PORT, IB_CLIENT_ID, TICKER, INTERVAL, 
                   MODEL_FILE, STOP_LOSS_PCT, POSITION_PCT)
from utils import setup_logging, ensure_dirs
from pathlib import Path
import ta
import pandas as pd

LOG_DIR = Path("logs")
ensure_dirs(LOG_DIR)
logger = setup_logging(LOG_DIR, "ibkr_bot")

def connect_ib(max_retries=3):
    """Connect to IB with retry logic"""
    ib = IB()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to IB at {IB_HOST}:{IB_PORT} (attempt {attempt+1}/{max_retries})")
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            
            if not ib.isConnected():
                raise ConnectionError("Connection failed")
            
            logger.info("Connected to IB successfully")
            return ib
        
        except Exception as e:
            logger.error(f"Connection attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise ConnectionError("Could not connect to IB. Is TWS/IB Gateway running and API enabled?")

def get_latest_bars_ib(ib, contract, duration="5 D", bar_size="5 mins"):
    """Fetch historical bars from IB"""
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,  # Increased from 2D to 5D for more data
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        df = util.df(bars)
        
        if df.empty:
            logger.warning("No bars returned from IB")
        else:
            logger.info(f"Fetched {len(df)} bars from IB, latest: {df.index[-1]}")
        
        return df
    except Exception as e:
        logger.exception(f"Error fetching bars: {e}")
        return pd.DataFrame()

def compute_features_df(df):
    """Compute features on IB data"""
    df2 = df.copy()
    
    # Handle MultiIndex columns if present
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = df2.columns.get_level_values(0)
    
    # Ensure lowercase columns
    df2.columns = df2.columns.str.lower()
    
    df2["sma_short"] = df2["close"].rolling(window=10).mean()
    df2["sma_long"] = df2["close"].rolling(window=50).mean()
    df2["rsi"] = ta.momentum.RSIIndicator(df2["close"], window=14).rsi()
    df2["atr"] = ta.volatility.average_true_range(df2["high"], df2["low"], df2["close"], window=14)
    df2["ret_1"] = df2["close"].pct_change(1)
    df2["ret_3"] = df2["close"].pct_change(3)  # ADDED: Missing ret_3
    
    for lag in range(1,4):
        df2[f"ret_lag_{lag}"] = df2["ret_1"].shift(lag)
    
    df2 = df2.dropna()
    return df2

def get_current_position(ib, contract):
    """Get actual position from IB"""
    positions = ib.positions()
    for pos in positions:
        if pos.contract.symbol == contract.symbol:
            return pos.position  # Returns quantity (positive = long, negative = short)
    return 0

def run_ibkr(dry_run=True, poll_seconds=60):
    """Main IBKR bot loop"""
    # Load model
    try:
        bundle = joblib.load(MODEL_FILE)
        model = bundle["model"]
        scaler = bundle["scaler"]
        features = bundle["features"]
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model")
        return

    # Connect to IB
    try:
        ib = connect_ib()
    except Exception as e:
        logger.exception("Failed to connect to IB")
        return
    
    contract = Stock(TICKER, "SMART", "USD")
    ib.qualifyContracts(contract)
    
    logger.info(f"Starting IBKR bot (dry_run={dry_run})")
    logger.info(f"Polling every {poll_seconds} seconds")
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while True:
        try:
            # Check connection
            if not ib.isConnected():
                logger.warning("Lost connection to IB, reconnecting...")
                ib = connect_ib()
                consecutive_errors = 0
            
            # Get recent bars
            df = get_latest_bars_ib(ib, contract, duration="2 D", bar_size="5 mins")
            if df.empty:
                logger.warning("No bars returned")
                time.sleep(poll_seconds)
                continue
            
            # Compute features
            df = compute_features_df(df)
            if df.empty:
                logger.warning("Not enough data for features")
                time.sleep(poll_seconds)
                continue
            
            last = df.iloc[-1]
            
            # Get prediction
            X = last[features].values.reshape(1, -1)
            Xs = scaler.transform(X)
            prob = model.predict_proba(Xs)[0,1]
            sig = int(prob > 0.6)
            action = "BUY" if sig==1 else "SELL"
            price = float(last["close"])
            
            logger.info(f"Decision: {action} prob={prob:.3f} price={price:.2f}")
            
            # Get current position from IB
            current_qty = get_current_position(ib, contract)
            logger.info(f"Current position: {current_qty} shares")
            
            if dry_run:
                logger.info("Dry-run mode: no order placed.")
            else:
                # Calculate order size based on capital
                account_value = [v for v in ib.accountValues() if v.tag == 'NetLiquidation'][0]
                capital = float(account_value.value)
                target_value = capital * POSITION_PCT
                target_qty = int(target_value / price)
                
                logger.info(f"Account value: ${capital:.2f}, Target position: {target_qty} shares")
                
                # Trading logic
                if sig == 1 and current_qty <= 0:
                    # Buy signal and not long
                    qty = max(1, target_qty)
                    order = MarketOrder("BUY", qty)
                    trade = ib.placeOrder(contract, order)
                    logger.info(f"Placed BUY order for {qty} shares: {trade}")
                    ib.sleep(2)  # Wait for order to fill
                    
                elif sig == 0 and current_qty > 0:
                    # Sell signal and currently long
                    order = MarketOrder("SELL", current_qty)
                    trade = ib.placeOrder(contract, order)
                    logger.info(f"Placed SELL order for {current_qty} shares: {trade}")
                    ib.sleep(2)
            
            consecutive_errors = 0  # Reset on success
            
        except Exception as e:
            consecutive_errors += 1
            logger.exception(f"IBKR loop error ({consecutive_errors}/{max_consecutive_errors})")
            
            if consecutive_errors >= max_consecutive_errors:
                logger.error("Too many consecutive errors, exiting")
                break
        
        time.sleep(poll_seconds)
    
    # Cleanup
    try:
        ib.disconnect()
        logger.info("Disconnected from IB")
    except Exception:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", default=True, help="Dry-run mode: no orders")
    parser.add_argument("--live", action="store_true", help="LIVE TRADING MODE (overrides --dry)")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()
    
    # Safety check: require explicit --live flag
    if args.live:
        response = input("WARNING: You are about to start LIVE TRADING. Type 'YES' to confirm: ")
        if response != "YES":
            print("Aborted.")
            exit()
        dry_run = False
    else:
        dry_run = True
    
    run_ibkr(dry_run=dry_run, poll_seconds=args.interval)