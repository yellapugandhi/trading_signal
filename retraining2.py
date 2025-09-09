import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from data import load_data, create_master_dataframe

import warnings
warnings.filterwarnings('ignore')


class AntiOverfittingModelTrainer:
    """🛡️ AntiOverfitting Trainer with Realistic Performance"""

    def __init__(self, symbol="NIFTY"):
        self.symbol = symbol
        self.models_dir = "models"
        self.reports_dir = "training_reports"
        self.df = None
        self.best_buy_model = None
        self.best_rr_model = None
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_and_prepare_data(self, days_back=360):
        print(f"📊 Loading data for symbol: {self.symbol} ...")
        try:
            grow, dfs = load_data(self.symbol, days_back)
            if not dfs:
                raise ValueError("No data retrieved from API")
            self.df = create_master_dataframe(dfs)
            if self.df.empty:
                raise ValueError("Empty dataframe after aggregation")
            print(f"✅ Loaded {len(self.df)} records from {self.symbol}")
            print(f"Date range: {self.df['timestamp'].min()} - {self.df['timestamp'].max()}")
            return True
        except Exception as e:
            print(f"❌ Failed to load  {e}")
            return False

    def enhanced_feature_engineering(self):
        print("🛠️ Starting feature engineering ...")
        try:
            df = self.df

            # Basic SMA, EMA features
            df["SMA_10"] = df["close"].rolling(10).mean()
            df["SMA_20"] = df["close"].rolling(20).mean()
            df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
            df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

            # Momentum and volatility
            df["Momentum"] = df["close"].diff(5)
            df["Momentum_10"] = df["close"].diff(10)
            df["Volatility"] = df["close"].rolling(10).std()
            df["Volatility_20"] = df["close"].rolling(20).std()

            # RSI
            df["RSI"] = self.compute_rsi(df["close"])

            # MACD + Signal + Histogram
            df["MACD"], df["MACD_Signal"] = self.compute_macd(df["close"])
            df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

            # Price positioning to SMA
            df["Price_Above_SMA10"] = (df["close"] > df["SMA_10"]).astype(int)
            df["Price_Above_SMA20"] = (df["close"] > df["SMA_20"]).astype(int)

            # Lagged features for safe history
            df["Lag_Close"] = df["close"].shift(1)
            df["Lag_Momentum"] = df["Momentum"].shift(1)
            df["Lag_RSI"] = df["RSI"].shift(1)

            # Price action: higher highs/lower lows
            df["Higher_High"] = (df["high"] > df["high"].shift(1)).astype(int)
            df["Lower_Low"] = (df["low"] < df["low"].shift(1)).astype(int)

            # Bollinger Bands
            df["BB_Middle"] = df["close"].rolling(20).mean()
            stddev = df["close"].rolling(20).std()
            df["BB_Upper"] = df["BB_Middle"] + 2 * stddev
            df["BB_Lower"] = df["BB_Middle"] - 2 * stddev
            df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
            df["BB_Position"] = (df["close"] - df["BB_Lower"]) / (df["BB_Width"])

            # Money Flow Index
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            money_flow = typical_price * df["volume"]
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
            pos_mf_sum = positive_flow.rolling(14).sum()
            neg_mf_sum = negative_flow.rolling(14).sum()
            df["MFI"] = 100 - 100 / (1 + pos_mf_sum / (neg_mf_sum + 1e-8))

            # Stochastic Oscillator
            low14 = df["low"].rolling(14).min()
            high14 = df["high"].rolling(14).max()
            df["Stoch_K"] = 100 * (df["close"] - low14) / (high14 - low14)
            df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

            # Williams %R
            df["Williams_R"] = -100 * (high14 - df["close"]) / (high14 - low14)

            # On Balance Volume
            obv = [0]
            for i in range(1, len(df)):
                if df.iloc[i]["close"] > df.iloc[i - 1]["close"]:
                    obv.append(obv[-1] + df.iloc[i]["volume"])
                elif df.iloc[i]["close"] < df.iloc[i - 1]["close"]:
                    obv.append(obv[-1] - df.iloc[i]["volume"])
                else:
                    obv.append(obv[-1])
            df["OBV"] = obv

            # Commodity Channel Index
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            df["CCI"] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * mad)

            # ADX
            try:
                tr1 = df["high"] - df["low"]
                tr2 = abs(df["high"] - df["close"].shift())
                tr3 = abs(df["low"] - df["close"].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                plus_dm = df["high"].diff()
                minus_dm = (-df["low"].diff())
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm < 0] = 0
                atr = tr.rolling(14).mean()
                plus_di = 100 * plus_dm.rolling(14).mean() / atr
                minus_di = 100 * minus_dm.rolling(14).mean() / atr
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                df["ADX"] = dx.rolling(14).mean()
            except:
                df["ADX"] = 25

            # Market Regimes
            df["Volatility"] = df["close"].rolling(10).std()
            df["Trend_Strength"] = df["SMA_20"].pct_change(5).abs()
            df["Volatility_Ratio"] = df["Volatility"] / (df["Volatility"].rolling(30).mean() + 1e-8)

            df["Market_Regime"] = "Normal"
            df.loc[df["Volatility_Ratio"] > 1.3, "Market_Regime"] = "High_Volatility"
            df.loc[df["Trend_Strength"] > 0.02, "Market_Regime"] = "Trending"
            df.loc[df["Trend_Strength"] <= 0.01, "Market_Regime"] = "Sideways"

            regime_map = {"Normal": 0, "High_Volatility": 1, "Trending": 2, "Sideways": 3}
            df["Market_Regime_Num"] = df["Market_Regime"].map(regime_map)

            self.df = df.reset_index(drop=True)
            print(f"Feature engineering complete, shape: {self.df.shape}")
            return True
        except Exception as e:
            print("Error in feature engineering:", e)
            return False

    def compute_rsi(self, price, period=14):
        delta = price.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta).clip(lower=0).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - 100 / (1 + rs)

    def compute_macd(self, price, n_fast=12, n_slow=26, n_signal=9):
        ema_fast = price.ewm(span=n_fast, adjust=False).mean()
        ema_slow = price.ewm(span=n_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=n_signal, adjust=False).mean()
        return macd, signal

    def create_targets(self):
        print("Creating targets...")
        df = self.df

        cond1 = (df["RSI"] < 45) & (df["Momentum"] > 0)
        cond2 = (df["close"] > df["SMA_10"])
        cond3 = (df["MACD"] > df["MACD_Signal"]) & (df["MACD_Histogram"] > 0)
        cond4 = (df["Momentum"] > df["Momentum"].quantile(0.6)) & (df["RSI"] < 70)

        cond5 = (df["MFI"] < 30) & (df["OBV"] > df["OBV"].shift(5))
        cond6 = (df["BB_Position"] < 0.2) & (df["close"] > df["close"].shift(1))
        cond7 = (df["Stoch_K"] < 20) & (df["Stoch_K"] > df["Stoch_D"])
        cond8 = (df["Williams_R"] < -80) & (df["Williams_R"] > df["Williams_R"].shift(1))
        cond9 = (df["CCI"] < -100) & (df["CCI"] > df["CCI"].shift(1))
        cond10 = (df["ADX"] > 25) & (df["close"] > df["SMA_10"])

        df["Buy_Score"] = (
            cond1.astype(float) * 0.35 +
            cond2.astype(float) * 0.25 +
            cond3.astype(float) * 0.25 +
            cond4.astype(float) * 0.15
        )

        df["Enhanced_Buy_Score"] = (
            cond1.astype(float) * 0.20 +
            cond2.astype(float) * 0.15 +
            cond3.astype(float) * 0.15 +
            cond4.astype(float) * 0.10 +
            cond5.astype(float) * 0.10 +
            cond6.astype(float) * 0.08 +
            cond7.astype(float) * 0.08 +
            cond8.astype(float) * 0.06 +
            cond9.astype(float) * 0.04 +
            cond10.astype(float) * 0.04
        )

        np.random.seed(42)
        noise = np.random.normal(1.0, 0.03, len(df))
        weighted_score = df["Buy_Score"] * noise

        threshold = 0.7
        df["Buy_Signal"] = (weighted_score > threshold).astype(int)

        future_returns = []
        min_gain = 0.015

        for i in range(len(df)):
            if i + 5 < len(df):
                future_high = df.loc[i+1:i+5, "high"].max()
                future_low = df.loc[i+1:i+5, "low"].min()
                curr_price = df.loc[i, "close"]
                if curr_price > 0:
                    gain = (future_high - curr_price) / curr_price
                    loss = (curr_price - future_low) / curr_price
                    if gain < min_gain:
                        rr = 0.05
                    elif loss > 0.002:
                        rr = gain / loss
                    else:
                        rr = gain * 30
                    rr *= (1 + df.loc[i, "Buy_Score"] * 0.05)
                    rr = max(min(rr, 3), 0.05)
                else:
                    rr = 0.05
                future_returns.append(rr)
            else:
                future_returns.append(np.mean(future_returns[-20:]) if len(future_returns) >= 20 else 1.0)

        df["Risk_Reward"] = future_returns
        df["Risk_Reward"] = df["Risk_Reward"].clip(0.05, 3)
        df["Risk_Reward"].fillna(1, inplace=True)

        self.df = df.reset_index(drop=True)

        print(f"Targets created: {df['Buy_Signal'].sum()} buys ({df['Buy_Signal'].mean()*100:.2f}%)")
        return True

    def balance_dataset(self):
        print("Balancing dataset...")
        df = self.df.dropna(subset=["Buy_Signal"])
        buy_df = df[df["Buy_Signal"] == 1]
        sell_df = df[df["Buy_Signal"] == 0]
        n = min(len(buy_df), len(sell_df))
        if n == 0:
            print("Cannot balance: insufficient class samples!")
            return False
        balanced_df = pd.concat([buy_df.sample(n), sell_df.sample(n)]).sample(frac=1).reset_index(drop=True)
        self.df = balanced_df
        print(f"Balanced dataset: {n} buys and {n} sells")
        return True

    def prepare_training_data(self):
        safe_features = [f for f in [
            "SMA_10", "EMA_10", "RSI", "Momentum", "Volatility", "Lag_Close", "Lag_Momentum",
            "MACD", "MACD_Signal", "BB_Position", "MFI", "Stoch_K", "Williams_R", "CCI",
            "ADX", "Volatility_Ratio", "Trend_Strength", "Market_Regime_Num"
        ] if f in self.df.columns]

        df = self.df.dropna(subset=safe_features + ["Buy_Signal", "Risk_Reward"])
        for f in safe_features:
            df[f].fillna(df[f].median(), inplace=True)
        df["Buy_Signal"].fillna(0, inplace=True)
        df["Risk_Reward"].fillna(1, inplace=True)

        X = df[safe_features]
        y_buy = df["Buy_Signal"]
        y_rr = df["Risk_Reward"]

        return X, y_buy, y_rr

    def train_buy_model(self, X, y):
        print("Training buy model...")
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 4],
            "min_samples_leaf": [20, 50],
            "min_samples_split": [50, 100],
            "max_features": ["sqrt", 0.5]
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True, class_weight='balanced')
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(rf, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X, y)
        self.best_buy_model = grid.best_estimator_

        print(f"Best buy model params: {grid.best_params_}")

    def train_rr_model(self, X, y):
        print("Training risk-reward model...")
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 4],
            "min_samples_leaf": [20, 50],
            "min_samples_split": [50, 100],
            "max_features": ["sqrt", 0.5]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
        grid.fit(X, y)
        self.best_rr_model = grid.best_estimator_

        print(f"Best RR model params: {grid.best_params_}")

    def save_models(self):
        if self.best_buy_model:
            path = os.path.join(self.models_dir, "buy_model.pkl")
            joblib.dump(self.best_buy_model, path)
            print(f"Saved buy model to {path}")
        if self.best_rr_model:
            path = os.path.join(self.models_dir, "rr_model.pkl")
            joblib.dump(self.best_rr_model, path)
            print(f"Saved RR model to {path}")

if __name__ == "__main__":
    trainer = AntiOverfittingModelTrainer()
    if trainer.load_and_prepare_data():
        if trainer.enhanced_feature_engineering():
            if trainer.create_targets():
                if trainer.balance_dataset():
                    X, y_buy, y_rr = trainer.prepare_training_data()
                    trainer.train_buy_model(X, y_buy)
                    trainer.train_rr_model(X, y_rr)
                    trainer.save_models()
                    print("Training complete!")
                else:
                    print("Failed to balance dataset.")
            else:
                print("Failed to create targets.")
        else:
            print("Feature engineering failed.")
    else:
        print("Data loading failed.")
