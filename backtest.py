"""
backtest.py
Runs a simple Backtrader backtest of SMA+RSI baseline and optional ML-filtered strategy.
Saves simple JSON results into results/.
"""
import backtrader as bt
import pandas as pd
import json
from pathlib import Path
from config import (CLEAN_DIR, RESULTS_DIR, TICKER, INTERVAL, INITIAL_CAPITAL, 
                   COMMISSION, POSITION_PCT, MODEL_FILE, STOP_LOSS_PCT, TAKE_PROFIT_PCT)
from utils import ensure_dirs, setup_logging
import joblib

ensure_dirs(RESULTS_DIR)
logger = setup_logging(RESULTS_DIR, "backtest")

class SmaRsiStrategy(bt.Strategy):
    params = dict(
        sma_short=10, 
        sma_long=50, 
        rsi_period=14,
        stop_loss=STOP_LOSS_PCT,
        take_profit=TAKE_PROFIT_PCT
    )

    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_short)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_long)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.order = None
        self.entry_price = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                logger.info(f"BUY EXECUTED at {self.entry_price:.2f}")
            elif order.issell():
                logger.info(f"SELL EXECUTED at {order.executed.price:.2f}")
                self.entry_price = None
        self.order = None

    def next(self):
        # Check stop-loss and take-profit
        if self.position:
            current_price = self.data.close[0]
            if self.entry_price:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                
                if pnl_pct <= -self.p.stop_loss:
                    logger.info(f"STOP LOSS triggered at {pnl_pct*100:.2f}%")
                    self.close()
                    return
                elif pnl_pct >= self.p.take_profit:
                    logger.info(f"TAKE PROFIT triggered at {pnl_pct*100:.2f}%")
                    self.close()
                    return
        
        # Entry logic
        if not self.position:
            if self.sma_short[0] > self.sma_long[0] and self.rsi[0] < 70:
                size = int((self.broker.getvalue() * POSITION_PCT) / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Exit on bearish cross
            if self.sma_short[0] < self.sma_long[0]:
                self.close()

class MLFilteredStrategy(SmaRsiStrategy):
    def __init__(self):
        super().__init__()
    
    def next(self):
        # read ml_signal if present (0/1) as a line in data
        ml_sig = None
        try:
            ml_sig = int(self.data.ml_signal[0])
        except Exception:
            ml_sig = None
        
        if ml_sig is not None:
            # Check stop-loss and take-profit first
            if self.position:
                current_price = self.data.close[0]
                if self.entry_price:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                    
                    if pnl_pct <= -self.p.stop_loss:
                        logger.info(f"STOP LOSS triggered at {pnl_pct*100:.2f}%")
                        self.close()
                        return
                    elif pnl_pct >= self.p.take_profit:
                        logger.info(f"TAKE PROFIT triggered at {pnl_pct*100:.2f}%")
                        self.close()
                        return
            
            # ML-filtered entry
            if not self.position:
                if ml_sig == 1 and self.sma_short[0] > self.sma_long[0] and self.rsi[0] < 70:
                    size = int((self.broker.getvalue() * POSITION_PCT) / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
            else:
                # Exit on ML sell signal or bearish cross
                if ml_sig == 0 or self.sma_short[0] < self.sma_long[0]:
                    self.close()
        else:
            # fallback to baseline
            super().next()

def run_baseline(ticker=TICKER, interval=INTERVAL):
    path = CLEAN_DIR / f"{ticker}_{interval}_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()
    logger.info(f"Running baseline backtest on {len(df)} bars")
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION)
    
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SmaRsiStrategy)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    
    logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    logger.info(f"Final Portfolio Value: {final_value:.2f}")
    
    analysis = {
        "initial_value": INITIAL_CAPITAL,
        "final_value": final_value,
        "total_return": ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100,
        "sharpe": strat.analyzers.sharpe.get_analysis(),
        "drawdown": strat.analyzers.drawdown.get_analysis(),
        "returns": strat.analyzers.returns.get_analysis(),
        "trades": strat.analyzers.trades.get_analysis()
    }
    
    out = RESULTS_DIR / f"backtest_baseline_{ticker}_{interval}.json"
    out.write_text(json.dumps(analysis, default=str, indent=2))
    logger.info(f"Wrote baseline results to {out}")
    
    try:
        cerebro.plot(style="candlestick")
    except Exception as e:
        logger.info(f"Plot not available: {e}")
    
    return out

def run_ml_backtest(ticker=TICKER, interval=INTERVAL, model_file=None, threshold=0.6):
    path = CLEAN_DIR / f"{ticker}_{interval}_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()
    
    if not model_file:
        model_file = MODEL_FILE
    
    logger.info(f"Loading model from {model_file}")
    model_bundle = joblib.load(model_file)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    features = model_bundle["features"]
    
    X = df[features].values
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:,1]
    df["ml_signal"] = (probs > threshold).astype(int)
    
    logger.info(f"ML Signals - BUY: {(df['ml_signal']==1).sum()}, SELL: {(df['ml_signal']==0).sum()}")
    
    # feed with ml_signal as an extra column
    class PandasWithSignal(bt.feeds.PandasData):
        lines = ("ml_signal",)
        params = (("ml_signal", -1),)
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION)
    
    data = PandasWithSignal(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(MLFilteredStrategy)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    
    logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    
    logger.info(f"Final Portfolio Value: {final_value:.2f}")
    
    analysis = {
        "initial_value": INITIAL_CAPITAL,
        "final_value": final_value,
        "total_return": ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100,
        "sharpe": strat.analyzers.sharpe.get_analysis(),
        "drawdown": strat.analyzers.drawdown.get_analysis(),
        "returns": strat.analyzers.returns.get_analysis(),
        "trades": strat.analyzers.trades.get_analysis(),
        "threshold": threshold
    }
    
    out = RESULTS_DIR / f"backtest_ml_{ticker}_{interval}_thresh{threshold}.json"
    out.write_text(json.dumps(analysis, default=str, indent=2))
    logger.info(f"Wrote ML backtest results to {out}")
    
    try:
        cerebro.plot(style="candlestick")
    except Exception as e:
        logger.info(f"Plot not available: {e}")
    
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline","ml","both"], default="baseline")
    parser.add_argument("--model", default=None)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    
    if args.mode == "baseline":
        run_baseline()
    elif args.mode == "ml":
        run_ml_backtest(model_file=args.model, threshold=args.threshold)
    else:  # both
        run_baseline()
        run_ml_backtest(model_file=args.model, threshold=args.threshold)