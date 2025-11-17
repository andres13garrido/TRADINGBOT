# Complete Testing Guide for Trading Bot

## Prerequisites

### 1. Install Required Packages
```bash
pip install yfinance pandas numpy scikit-learn ta backtrader joblib ib_insync matplotlib
```

### 2. Project Structure
Create this folder structure:
```
trading_bot/
├── config.py
├── utils.py
├── fetch_data.py
├── features.py
├── train_model.py
├── backtest.py
├── dry_run_bot.py
├── ibkr_bot.py
├── data/
│   ├── raw/
│   └── cleaned/
├── models/
├── results/
└── logs/
```

---

## Step-by-Step Testing

### STEP 1: Fetch Historical Data (5 minutes)

```bash
python fetch_data.py
```

**What it does:**
- Downloads SPY daily data from 2018-01-01 to today
- Saves to `data/raw/SPY_1d.csv`

**Expected output:**
```
INFO Fetching SPY interval=1d start=2018-01-01 end=None
INFO Saved raw CSV to data/raw/SPY_1d.csv with 1729 rows
```

**Verify:**
```bash
# Check file exists and has data
head data/raw/SPY_1d.csv
```

---

### STEP 2: Compute Technical Features (2 minutes)

```bash
python features.py
```

**What it does:**
- Loads raw CSV
- Computes SMA, RSI, ATR, lagged returns
- Saves to `data/cleaned/SPY_1d_features.csv`

**Expected output:**
```
INFO Loaded 1729 rows from data/raw/SPY_1d.csv
INFO Computing technical indicators...
INFO Features computed, 1668 rows after dropna
INFO Saved features to data/cleaned/SPY_1d_features.csv
```

**Verify:**
```bash
# Check feature columns
head -1 data/cleaned/SPY_1d_features.csv
```

You should see columns: `open, high, low, close, volume, sma_short, sma_long, rsi, atr, ret_1, ret_3, ret_lag_1, ret_lag_2, ret_lag_3`

---

### STEP 3: Train ML Model (30 seconds)

```bash
python train_model.py
```

**What it does:**
- Creates binary labels (up/down next day)
- Trains Random Forest on 80% of data
- Tests on remaining 20%
- Saves model to `models/rf_model.joblib`

**Expected output:**
```
INFO Loaded 1668 rows for training
INFO Class distribution - 0: 807, 1: 861
INFO Train size: 1334, Test size: 334
INFO Training Random Forest...
INFO Validation accuracy: 0.5269
INFO 
              precision    recall  f1-score   support

           0       0.53      0.54      0.53       160
           1       0.53      0.52      0.52       174

    accuracy                           0.53       334
   macro avg       0.53      0.53      0.53       334
weighted avg       0.53      0.53      0.53       334

INFO Feature Importance:
     feature  importance
0        rsi    0.186422
1  sma_short    0.165338
2   sma_long    0.146829
...
INFO Saved model to models/rf_model.joblib
```

**Important Notes:**
- Accuracy around 52-54% is NORMAL for financial markets (barely better than coin flip)
- We're looking for edge, not perfection
- Feature importance shows which indicators matter most

**Verify:**
```bash
ls -lh models/rf_model.joblib
```

---

### STEP 4: Run Baseline Backtest (10 seconds)

```bash
python backtest.py --mode baseline
```

**What it does:**
- Tests pure technical strategy (SMA + RSI)
- No ML filtering
- Simulates trading with $100k capital, 0.05% commission

**Expected output:**
```
INFO Running baseline backtest on 1668 bars
INFO Starting Portfolio Value: 100000.00
INFO BUY EXECUTED at 267.23
INFO SELL EXECUTED at 276.45
...
INFO Final Portfolio Value: 103456.78
INFO Wrote baseline results to results/backtest_baseline_SPY_1d.json
```

**Analyze Results:**
```bash
cat results/backtest_baseline_SPY_1d.json
```

**Key metrics to look for:**
```json
{
  "initial_value": 100000,
  "final_value": 103456.78,
  "total_return": 3.46,
  "sharpe": {"sharperatio": 0.45},
  "drawdown": {"max": {"drawdown": 12.3}},
  "trades": {
    "total": {"total": 34},
    "won": {"total": 18}
  }
}
```

**What's good:**
- Positive return
- Sharpe > 0.5 (decent risk-adjusted returns)
- Max drawdown < 20%
- Win rate around 50%

---

### STEP 5: Run ML-Enhanced Backtest (10 seconds)

```bash
python backtest.py --mode ml --threshold 0.6
```

**What it does:**
- Uses ML model to filter signals
- Only enters when model confidence > 60%
- Should reduce false signals

**Expected output:**
```
INFO Loading model from models/rf_model.joblib
INFO ML Signals - BUY: 687, SELL: 981
INFO Running ML backtest on 1668 bars
INFO Final Portfolio Value: 105234.12
```

**Compare to Baseline:**
```bash
# View both results side-by-side
cat results/backtest_baseline_SPY_1d.json | grep "total_return"
cat results/backtest_ml_SPY_1d_thresh0.6.json | grep "total_return"
```

**What to expect:**
- ML version might have higher return OR lower drawdown
- Fewer trades (more selective)
- Better risk-adjusted returns (higher Sharpe)

---

### STEP 6: Test Different ML Thresholds (30 seconds)

```bash
python backtest.py --mode ml --threshold 0.55
python backtest.py --mode ml --threshold 0.65
python backtest.py --mode ml --threshold 0.70
```

**Analysis:**
- Lower threshold (0.55): More trades, more risk
- Higher threshold (0.70): Fewer trades, more selective
- Find sweet spot with best Sharpe ratio

---

### STEP 7: Dry Run Bot (Safe Testing)

```bash
python dry_run_bot.py --interval 60
```

**What it does:**
- Fetches live data from yfinance every 60 seconds
- Computes features
- Makes prediction
- **LOGS decision but DOESN'T place orders**

**Expected output (repeating every 60 seconds):**
```
INFO Model loaded successfully
INFO Features: ['sma_short', 'sma_long', 'rsi', 'atr', 'ret_1', 'ret_3', 'ret_lag_1', 'ret_lag_2', 'ret_lag_3']
INFO Starting dry-run loop (no orders will be sent).
INFO Signal=BUY prob=0.623 price=456.78 time=2024-11-17 14:32:01
INFO Signal=BUY prob=0.618 price=456.82 time=2024-11-17 14:33:01
INFO Signal=SELL prob=0.542 price=456.45 time=2024-11-17 14:34:01
```

**Let it run for 10-15 minutes and observe:**
- Does it fetch data successfully?
- Are predictions reasonable (not all BUY or all SELL)?
- Check logs: `tail -f logs/dry_run_*.log`

**Stop with:** `Ctrl+C`

---

### STEP 8: IBKR Paper Trading (Advanced)

⚠️ **REQUIRES INTERACTIVE BROKERS ACCOUNT AND TWS/IB GATEWAY**

**Setup:**
1. Download IB Gateway or TWS
2. Enable API access (Configuration → API → Settings)
3. Set port to 7497 (paper trading)
4. Start TWS/Gateway

**Test connection:**
```bash
python ibkr_bot.py --dry --interval 60
```

**Expected output:**
```
INFO Connecting to IB at 127.0.0.1:7497
INFO Connected to IB successfully
INFO Starting IBKR bot (dry_run=True)
INFO Decision: BUY prob=0.623 price=456.78
INFO Current position: 0 shares
INFO Dry-run mode: no order placed.
```

**If you see connection errors:**
- Check TWS/Gateway is running
- Verify API is enabled
- Ensure port 7497 is correct
- Try restarting TWS

---

## Common Issues & Fixes

### Issue 1: "No data fetched"
```
RuntimeError: No data fetched.
```
**Fix:**
- Check internet connection
- Try different date range: `python fetch_data.py --start 2020-01-01`
- Use different ticker: `python fetch_data.py --ticker AAPL`

---

### Issue 2: "Missing features"
```
KeyError: Missing features: ['ret_lag_1']
```
**Fix:**
- Re-run features.py: `python features.py`
- Check cleaned CSV has all columns

---

### Issue 3: Low accuracy (< 50%)
```
INFO Validation accuracy: 0.4850
```
**This is actually normal!** Financial markets are noisy. Focus on:
- Positive returns in backtest
- Sharpe ratio > 0.5
- Drawdown < 20%

---

### Issue 4: Backtrader crashes
```
AttributeError: 'DataFrame' object has no attribute 'Close'
```
**Fix:**
- Already fixed in updated code (lowercase columns)
- If still happens: `df.columns = df.columns.str.lower()`

---

### Issue 5: IB connection fails
```
ConnectionError: Could not connect to IB
```
**Checklist:**
1. TWS/Gateway running?
2. API enabled? (Configuration → API → Settings)
3. Correct port? (7497 for paper, 7496 for live)
4. Firewall blocking connection?

---

## Performance Expectations

### Good Results:
✅ Baseline return: 2-5% over backtest period
✅ ML return: 3-7% (better than baseline)
✅ Sharpe ratio: 0.5-1.5
✅ Max drawdown: 10-20%
✅ Win rate: 48-55%

### Warning Signs:
⚠️ Return > 50%: Likely overfitting or bug
⚠️ Win rate > 70%: Lookahead bias
⚠️ Sharpe > 3: Too good to be true
⚠️ Zero drawdown: Not trading at all

---

## Next Steps After Testing

### 1. **Walk-Forward Optimization**
Retrain model every 3 months to adapt to market changes:
```bash
# Train on data up to June 2024
python train_model.py --end 2024-06-01

# Backtest on July-October 2024
python backtest.py --start 2024-07-01 --end 2024-10-31
```

### 2. **Parameter Tuning**
Edit `config.py` and test:
- `POSITION_PCT`: 0.005, 0.01, 0.02 (position size)
- `STOP_LOSS_PCT`: 0.01, 0.02, 0.03 (stop loss)
- `COMMISSION`: 0.0005, 0.001 (broker fees)

### 3. **Feature Engineering**
Add more indicators to `features.py`:
- MACD
- Bollinger Bands
- Volume indicators

### 4. **Model Improvements**
Try different algorithms in `train_model.py`:
- XGBoost
- LightGBM
- Ensemble (Random Forest + XGBoost)

### 5. **Paper Trading**
Let `dry_run_bot.py` run for 1-2 weeks and track decisions in a spreadsheet

---

## Safety Checklist Before Live Trading

❌ **DO NOT skip these steps:**

1. ✅ Backtest shows consistent profits over 2+ years
2. ✅ Paper trading for 30+ days with positive results
3. ✅ Understand every line of code
4. ✅ Have stop-loss protection (in config.py)
5. ✅ Start with tiny position size (0.001 of capital)
6. ✅ Monitor first 50 trades manually
7. ✅ Set maximum daily loss limit
8. ✅ Have kill switch ready (Ctrl+C stops bot)

---

## Questions to Answer Before Going Live

1. **What is your maximum acceptable loss?**
   - Set `STOP_LOSS_PCT` accordingly

2. **How much capital can you afford to lose entirely?**
   - Only trade with that amount

3. **Do you understand why the strategy works?**
   - Can you explain it in 2 sentences?

4. **What happens if internet drops for 1 hour?**
   - Bot will crash (add reconnection logic)

5. **What's your edge over random trading?**
   - Show me the backtest proving it

---

## Emergency Stop Procedures

### To stop dry_run_bot or ibkr_bot:
```bash
# Press Ctrl+C
# Or from another terminal:
ps aux | grep python
kill <PID>
```

### To close all IB positions manually:
1. Open TWS
2. Go to Portfolio
3. Right-click position → Close Position

---

## Final Recommendations

1. **Never skip backtesting** - If it doesn't work historically, it won't work live
2. **Paper trade for weeks** - Not days, WEEKS
3. **Start microscopically small** - 1% of your planned size
4. **Log everything** - You'll need it for debugging
5. **Expect losses** - Even good strategies lose 40-50% of trades
6. **Have an exit plan** - When do you stop? -10%? -20%?
7. **Markets change** - A strategy working now may fail next month

---

## Support & Resources

- **Backtrader Docs**: https://www.backtrader.com/docu/
- **ib_insync Docs**: https://ib-insync.readthedocs.io/
- **yfinance Issues**: https://github.com/ranaroussi/yfinance

Good luck, and remember: **Most traders lose money. Test thoroughly.**