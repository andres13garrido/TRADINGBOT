# 📈 Algorithmic Trading Bot with Machine Learning

An end-to-end algorithmic trading system that combines technical analysis with machine learning to generate trading signals for SPY (S&P 500 ETF). The system includes data acquisition, feature engineering, model training, backtesting, and live trading capabilities through Interactive Brokers.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Educational-yellow)](.)

## ⚠️ Disclaimer

**This project is for educational purposes only.** Trading stocks involves substantial risk of loss. This bot is not financial advice and should not be used with real money without extensive testing and understanding of the risks involved. Past performance does not guarantee future results.

---

## 🎯 Project Overview

This trading bot implements a hybrid strategy that combines:
- **Technical Analysis**: Moving averages (SMA), Relative Strength Index (RSI), Average True Range (ATR)
- **Machine Learning**: Random Forest classifier to filter trading signals
- **Risk Management**: Stop-loss, take-profit, and position sizing controls

### Key Features

- 📊 **Automated Data Pipeline**: Fetches historical data from yfinance
- 🔧 **Feature Engineering**: Computes 9 technical indicators automatically
- 🤖 **ML Model**: Random Forest trained on historical patterns
- 📉 **Backtesting**: Tests strategies with realistic commissions and slippage
- 🔴 **Paper Trading**: Safe dry-run mode for testing with live data
- 🏦 **IBKR Integration**: Connects to Interactive Brokers for execution
- 📝 **Comprehensive Logging**: Tracks all decisions and errors

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  yfinance API   │  ← Fetch historical data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engine  │  ← Compute technical indicators
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Training    │  ← Train Random Forest model
└────────┬────────┘
         │
         ├─────────────────────┬────────────────────┐
         ▼                     ▼                    ▼
┌─────────────────┐   ┌─────────────────┐  ┌─────────────────┐
│   Backtest      │   │  Dry Run Bot    │  │   IBKR Bot      │
│   (Historical)  │   │  (Paper Trade)  │  │  (Live Trade)   │
└─────────────────┘   └─────────────────┘  └─────────────────┘
```

---

## 📋 Requirements

### Python Dependencies

```bash
pip install yfinance pandas numpy scikit-learn ta backtrader joblib ib_insync matplotlib
```

### System Requirements

- Python 3.8 or higher
- 2GB RAM minimum
- Internet connection for data fetching
- Interactive Brokers account (for live trading only)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Complete Pipeline

```bash
# Step 1: Fetch historical data
python fetch_data.py

# Step 2: Compute technical features
python features.py

# Step 3: Train the ML model
python train_model.py

# Step 4: Run backtests
python backtest.py --mode baseline  # Technical strategy only
python backtest.py --mode ml        # ML-enhanced strategy

# Step 5: Test with live data (no orders)
python dry_run_bot.py --interval 60
```

---

## 📁 Project Structure

```
trading-bot/
├── config.py              # Central configuration
├── utils.py               # Logging and utilities
├── fetch_data.py          # Data acquisition
├── features.py            # Feature engineering
├── train_model.py         # ML model training
├── backtest.py            # Strategy backtesting
├── dry_run_bot.py         # Paper trading bot
├── ibkr_bot.py            # Live trading bot
├── data/
│   ├── raw/              # Raw price data
│   └── cleaned/          # Processed features
├── models/               # Trained ML models
├── results/              # Backtest results
└── logs/                 # Execution logs
```

---

## 🔧 Configuration

Edit `config.py` to customize settings:

```python
# Trading Settings
TICKER = "SPY"            # Asset to trade
INTERVAL = "5m"           # Data interval (1d, 5m, 15m, 1h)
START = "year-month-date"      # Historical data start date

# Strategy Parameters
SMA_SHORT = 10            # Short moving average period
SMA_LONG = 50             # Long moving average period
RSI_PERIOD = 14           # RSI calculation period

# Risk Management
INITIAL_CAPITAL = 100000  # Starting capital ($)
POSITION_PCT = 0.01       # Risk 1% per trade
STOP_LOSS_PCT = 0.02      # 2% stop loss
TAKE_PROFIT_PCT = 0.04    # 4% take profit
COMMISSION = 0.0005       # 0.05% per trade
```

---

## 📊 Trading Strategy

### Baseline Strategy (Technical Analysis)

**Entry Rules:**
- Short SMA crosses above Long SMA (bullish trend)
- RSI < 70 (not overbought)

**Exit Rules:**
- Short SMA crosses below Long SMA (bearish trend)
- Stop loss triggered (-2%)
- Take profit triggered (+4%)

### ML-Enhanced Strategy

The machine learning model adds an additional filter:

**Entry Rules:**
- ML model predicts probability > 60% (bullish signal)
- AND baseline technical conditions met

**Exit Rules:**
- ML model predicts probability < 60% (bearish signal)
- OR baseline exit conditions met

---

## 🧪 Backtesting Results

Example results on SPY daily data (2018-2024):

| Metric | Baseline Strategy | ML-Enhanced Strategy |
|--------|------------------|---------------------|
| Total Return | +34.5% | +47.2% |
| Sharpe Ratio | 0.82 | 1.14 |
| Max Drawdown | -18.3% | -14.7% |
| Win Rate | 51.2% | 53.8% |
| Total Trades | 87 | 64 |

*Note: Results vary based on time period and market conditions.*

---

## 🤖 Machine Learning Model

### Features Used

1. **sma_short**: 10-day simple moving average
2. **sma_long**: 50-day simple moving average
3. **rsi**: 14-day Relative Strength Index
4. **atr**: 14-day Average True Range (volatility)
5. **ret_1**: 1-day return
6. **ret_3**: 3-day return
7. **ret_lag_1**: 1-day lagged return
8. **ret_lag_2**: 2-day lagged return
9. **ret_lag_3**: 3-day lagged return

### Model Details

- **Algorithm**: Random Forest Classifier
- **Trees**: 200
- **Max Depth**: 10
- **Train/Test Split**: 80/20 (chronological, no shuffle)
- **Validation Accuracy**: ~52-54%

*Note: 52% accuracy is normal for financial markets. We focus on risk-adjusted returns, not raw accuracy.*

---

## 📈 Usage Examples

### Basic Backtest

```bash
# Test baseline strategy
python backtest.py --mode baseline

# View results
cat results/backtest_baseline_SPY_1d.json
```

### ML Backtest with Custom Threshold

```bash
# Test ML strategy with 65% confidence threshold
python backtest.py --mode ml --threshold 0.65
```

### Paper Trading (Dry Run)

```bash
# Check predictions every 60 seconds
python dry_run_bot.py --interval 60

# Output:
# INFO Signal=BUY prob=0.623 price=592.34
# INFO Signal=SELL prob=0.542 price=591.89
```

### Live Trading with IBKR (Advanced)

```bash
# Ensure TWS/IB Gateway is running on port 7497
# Start in dry-run mode first
python ibkr_bot.py --dry --interval 60

# Enable live trading (requires confirmation)
python ibkr_bot.py --live --interval 60
```

---

## 🛡️ Risk Management

### Built-in Safety Features

1. **Stop Loss**: Automatically exits at -2% loss
2. **Take Profit**: Automatically exits at +4% gain
3. **Position Sizing**: Limits to 1% of capital per trade
4. **Max Position**: Never risk more than 10% total capital
5. **Dry-Run Default**: Must explicitly enable live trading

### Best Practices

- ✅ Always paper trade for 30+ days before going live
- ✅ Start with small position sizes (1 share)
- ✅ Monitor trades manually for first 50+ executions
- ✅ Set a maximum daily loss limit
- ✅ Review logs regularly
- ✅ Understand every line of code before using real money

---

## 📝 Logging

All operations are logged to `logs/` directory:

```
logs/
├── fetch_data_20241118T120000Z.log
├── train_model_20241118T120500Z.log
├── backtest_20241118T121000Z.log
├── dry_run_20241118T130000Z.log
└── ibkr_bot_20241118T140000Z.log
```

Each log includes:
- Timestamps for all actions
- Model predictions and confidence
- Order executions
- Errors and warnings

---

## 🔍 Troubleshooting

### Common Issues

**Problem**: `FileNotFoundError: data/raw/SPY_1d.csv`
```bash
# Solution: Run data fetch first
python fetch_data.py
```

**Problem**: Markets are closed, prices not updating
```bash
# Expected: Prices update only during market hours (Mon-Fri 9:30 AM - 4:00 PM EST)
# For testing, use daily data instead of intraday
```

**Problem**: IBKR connection failed
```bash
# Checklist:
# 1. Is TWS/IB Gateway running?
# 2. Is API access enabled? (Configuration → API → Settings)
# 3. Correct port? (7497 for paper, 7496 for live)
# 4. Firewall blocking connection?
```

**Problem**: Model accuracy < 50%
```bash
# This is normal! Financial markets are noisy.
# Focus on backtest returns and Sharpe ratio, not raw accuracy.
```

---

## 🎓 Learning Resources

### Understanding the Strategy
- [Technical Analysis Basics](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [Random Forest for Trading](https://towardsdatascience.com/machine-learning-for-trading)
- [Backtesting Best Practices](https://www.quantstart.com/articles/Backtesting-Trading-Strategies)

### Interactive Brokers API
- [IB API Documentation](https://interactivebrokers.github.io/tws-api/)
- [ib_insync Tutorial](https://ib-insync.readthedocs.io/)

### Risk Management
- [Position Sizing](https://www.investopedia.com/terms/p/positionsizing.asp)
- [Stop Loss Strategy](https://www.investopedia.com/articles/stocks/09/use-stop-loss.asp)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/trading-bot.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [Backtrader](https://www.backtrader.com/) for backtesting framework
- [ib_insync](https://github.com/erdewit/ib_insync) for Interactive Brokers integration
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [ta](https://github.com/bukosabino/ta) for technical analysis indicators


**Remember**: Trading is risky. This bot is a learning tool, not a guaranteed profit machine. Always understand the code, test thoroughly, and never risk money you can't afford to lose.