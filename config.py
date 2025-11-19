from pathlib import Path

# ROOT is the TRADINGBOT folder (where config.py lives)
ROOT = Path(__file__).resolve().parent

# Data settings
TICKER = "SPY"
INTERVAL = "5m"            # yfinance interval
START = None
END = None                

# Features / Strategy
SMA_SHORT = 10
SMA_LONG = 50
RSI_PERIOD = 14

# Model
MODEL_FILE = ROOT / "models" / "rf_model.joblib"
ML_THRESHOLD = 0.52  # Probability threshold for ML signals (0.5-0.7 range)

# Backtest
INITIAL_CAPITAL = 100000
POSITION_PCT = 0.01
COMMISSION = 0.0005

# Risk Management
STOP_LOSS_PCT = 0.02       # 2% stop loss
TAKE_PROFIT_PCT = 0.04     # 4% take profit
MAX_POSITION_SIZE = 0.1    # Max 10% of capital per position

# Paths
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "cleaned"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"

# IBKR (paper default)
IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 1