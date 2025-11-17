"""
train_model.py
Trains RandomForestClassifier on features. Saves model to models/.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from config import CLEAN_DIR, MODELS_DIR, TICKER, INTERVAL, MODEL_FILE
from utils import ensure_dirs, setup_logging
import argparse

ensure_dirs(MODELS_DIR)
logger = setup_logging(MODELS_DIR, "train_model")

DEFAULT_FEATURES = ["sma_short","sma_long","rsi","atr","ret_1","ret_3","ret_lag_1","ret_lag_2","ret_lag_3"]

def main(ticker=TICKER, interval=INTERVAL, features=DEFAULT_FEATURES):
    path = CLEAN_DIR / f"{ticker}_{interval}_features.csv"
    if not path.exists():
        logger.error(f"Features CSV not found: {path}")
        raise FileNotFoundError(path)
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()
    logger.info(f"Loaded {len(df)} rows for training")
    
    # Create label: next period up/down
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()
    
    # Check for missing features
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        raise KeyError(f"Missing features: {missing}")
    
    X = df[features].copy()
    y = df["target"].copy()
    
    logger.info(f"Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()}")
    
    # time-ordered train/test split (no shuffle to prevent lookahead bias)
    split = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train model
    logger.info("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
    clf.fit(X_train_s, y_train)
    
    # Predictions
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:,1]
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    logger.info(f"Validation accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))
    logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(y_test, preds)))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info("\nFeature Importance:\n" + str(importance))
    
    # Save model + scaler together
    ensure_dirs(MODELS_DIR)
    joblib.dump({
        "model": clf, 
        "scaler": scaler, 
        "features": features,
        "accuracy": acc,
        "importance": importance.to_dict()
    }, MODEL_FILE)
    logger.info(f"Saved model to {MODEL_FILE}")
    return MODEL_FILE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument("--interval", default=INTERVAL)
    args = parser.parse_args()
    main(args.ticker, args.interval)