"""
ibkr_bot.py
Connects to Interactive Brokers via ib_insync, uses the trained model to make decisions,
and places orders if dry_run=False. ALWAYS test with dry_run=True first.
"""
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import joblib
import time
from config import (IB_HOST, IB_PORT, IB_CLIENT_ID, TICKER, INTERVAL, 
                   MODEL_FILE, STOP_LOSS_PCT, POSITION_PCT, ML_THRESHOLD)
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
        logger.info(f"ML Threshold: {ML_THRESHOLD}")
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
    
    # Track pending orders to prevent duplicates
    pending_order = None
    last_action = None
    
    while True:
        try:
            # Check connection
            if not ib.isConnected():
                logger.warning("Lost connection to IB, reconnecting...")
                ib = connect_ib()
                consecutive_errors = 0
            
            # Get recent bars
            df = get_latest_bars_ib(ib, contract, duration="5 D", bar_size="5 mins")
            if df.empty:
                logger.warning("No bars returned, waiting...")
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
            # Convert to DataFrame to preserve feature names (silences sklearn warning)
            X_df = pd.DataFrame(X, columns=features)
            Xs = scaler.transform(X_df)
            prob = model.predict_proba(Xs)[0,1]
            sig = int(prob > ML_THRESHOLD)
            action = "BUY" if sig==1 else "SELL"
            price = float(last["close"])
            
            logger.info(f"Decision: {action} prob={prob:.3f} price={price:.2f} (threshold={ML_THRESHOLD})")
            
            # Get current position from IB
            current_qty = get_current_position(ib, contract)
            logger.info(f"Current position: {current_qty} shares")
            
            if dry_run:
                logger.info("Dry-run mode: no order placed.")
            else:
                # Wait for pending orders to settle
                if pending_order and not pending_order.isDone():
                    logger.info("⏳ Waiting for previous order to complete...")
                    ib.sleep(2)
                    
                # Get fresh position after waiting
                current_qty = get_current_position(ib, contract)
                logger.info(f"Updated position after wait: {current_qty} shares")
                
                # Trading logic - only trade if no pending order
                if sig == 1 and current_qty == 0 and last_action != "BUY":
                    # Buy signal, no position, and haven't just bought
                    qty = 1
                    order = MarketOrder("BUY", qty)
                    trade = ib.placeOrder(contract, order)
                    logger.info(f"✅ PLACED BUY ORDER for {qty} shares: {trade}")
                    pending_order = trade
                    last_action = "BUY"
                    ib.sleep(2)  # Wait for order to fill
                    
                elif sig == 0 and current_qty > 0 and last_action != "SELL":
                    # Sell signal, have position, and haven't just sold
                    order = MarketOrder("SELL", current_qty)
                    trade = ib.placeOrder(contract, order)
                    logger.info(f"✅ PLACED SELL ORDER for {current_qty} shares: {trade}")
                    pending_order = trade
                    last_action = "SELL"
                    ib.sleep(2)
                else:
                    logger.info(f"No action: sig={sig}, qty={current_qty}, last={last_action}")
            
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
    parser.add_argument("--paper", action="store_true", help="Paper trading mode: places 1-share orders")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()
    
    # Safety check: require explicit --live flag
    if args.live:
        response = input("WARNING: You are about to start LIVE TRADING. Type 'YES' to confirm: ")
        if response != "YES":
            print("Aborted.")
            exit()
        dry_run = False
    elif args.paper:
        print("PAPER TRADING MODE: Will place 1-share orders visible in IB")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            exit()
        dry_run = False  # Enable order placement
    else:
        dry_run = True
    
    run_ibkr(dry_run=dry_run, poll_seconds=args.interval)