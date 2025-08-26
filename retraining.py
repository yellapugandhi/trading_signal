import os
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, mean_squared_error, classification_report, 
    roc_auc_score, confusion_matrix
)
from data import load_data, create_master_dataframe

warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self, symbol="NIFTY"):
        self.symbol = symbol
        self.models_dir = "models"
        self.reports_dir = "training_reports"
        self.df = None
        self.best_buy_model = None
        self.best_rr_model = None
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def load_and_prepare_data(self, days_back=360):
        """Load and prepare training data"""
        print(f"📊 Loading data for {self.symbol}...")
        
        try:
            groww_api, data_frames = load_data(self.symbol, days_back)
            
            if not data_frames:
                raise ValueError("No data frames retrieved")
            
            # Create master dataframe
            self.df = create_master_dataframe(data_frames)
            
            if self.df.empty:
                raise ValueError("Empty master dataframe")
            
            print(f"✅ Loaded {len(self.df)} data points")
            print(f"📅 Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading  {e}")
            return False
    
    def feature_engineering(self):
        """Apply comprehensive feature engineering"""
        print("🔧 Applying feature engineering...")
        
        try:
            # Technical indicators
            self.df["SMA_10"] = self.df["close"].rolling(10).mean()
            self.df["SMA_20"] = self.df["close"].rolling(20).mean()
            self.df["EMA_10"] = self.df["close"].ewm(span=10, adjust=False).mean()
            self.df["EMA_20"] = self.df["close"].ewm(span=20, adjust=False).mean()
            
            # Price momentum and volatility
            self.df["Momentum"] = self.df["close"] - self.df["close"].shift(5)
            self.df["Momentum_10"] = self.df["close"] - self.df["close"].shift(10)
            self.df["Volatility"] = self.df["close"].rolling(10).std()
            self.df["Volatility_20"] = self.df["close"].rolling(20).std()
            
            # RSI calculation
            self.df["RSI"] = self.compute_rsi(self.df["close"])
            
            # MACD calculation
            self.df["MACD"], self.df["MACD_Signal"] = self.compute_macd(self.df["close"])
            self.df["MACD_Histogram"] = self.df["MACD"] - self.df["MACD_Signal"]
            
            # Price position indicators
            self.df["Price_Above_SMA10"] = (self.df["close"] > self.df["SMA_10"]).astype(int)
            self.df["Price_Above_SMA20"] = (self.df["close"] > self.df["SMA_20"]).astype(int)
            
            # Lagged features
            self.df["Lag_Close"] = self.df["close"].shift(1)
            self.df["Lag_Volume"] = self.df["volume"].shift(1)
            self.df["Lag_Momentum"] = self.df["Momentum"].shift(1)
            self.df["Lag_RSI"] = self.df["RSI"].shift(1)
            
            # Volume indicators
            self.df["Volume_SMA"] = self.df["volume"].rolling(20).mean()
            self.df["Volume_Ratio"] = self.df["volume"] / self.df["Volume_SMA"].replace(0, 1)
            
            # Price patterns
            self.df["Higher_High"] = (self.df["high"] > self.df["high"].shift(1)).astype(int)
            self.df["Lower_Low"] = (self.df["low"] < self.df["low"].shift(1)).astype(int)
            
            print(f"✅ Feature engineering completed. DataFrame shape: {self.df.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Error in feature engineering: {e}")
            return False
    
    def compute_rsi(self, series, period=14):
        """Compute RSI indicator"""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def compute_macd(self, series, span1=12, span2=26, span_signal=9):
        """Compute MACD indicator"""
        ema1 = series.ewm(span=span1, adjust=False).mean()
        ema2 = series.ewm(span=span2, adjust=False).mean()
        macd = ema1 - ema2
        signal = macd.ewm(span=span_signal, adjust=False).mean()
        return macd, signal
    
    def create_targets(self):
        """Create target variables for training"""
        print("🎯 Creating target variables...")
        
        try:
            # **FIXED: More realistic buy signal conditions**
            # Create multiple buy signal conditions and combine them
            
            # Condition 1: RSI oversold with positive momentum
            cond1 = (self.df["RSI"] < 45) & (self.df["Momentum"] > 0)
            
            # Condition 2: Price above short-term moving average with good volume
            cond2 = (self.df["close"] > self.df["SMA_10"]) & (self.df["Volume_Ratio"] > 1.1)
            
            # Condition 3: MACD bullish crossover
            cond3 = (self.df["MACD"] > self.df["MACD_Signal"]) & (self.df["MACD_Histogram"] > 0)
            
            # Condition 4: Price momentum with RSI not overbought
            cond4 = (self.df["Momentum"] > self.df["Momentum"].quantile(0.6)) & (self.df["RSI"] < 70)
            
            # **KEY FIX: Use OR logic instead of AND to get more signals**
            self.df["Buy_Signal"] = (cond1 | cond2 | cond3 | cond4).astype(int)
            
            # **FIXED: Simplified risk-reward calculation**
            # Use next 3-day high/low for more stable calculation
            future_returns = []
            for i in range(len(self.df)):
                if i < len(self.df) - 3:
                    # Look ahead 3 periods instead of 5
                    future_high = self.df["high"].iloc[i+1:i+4].max()
                    future_low = self.df["low"].iloc[i+1:i+4].min()
                    current_price = self.df["close"].iloc[i]
                    
                    if current_price > 0:
                        potential_gain = (future_high - current_price) / current_price
                        potential_loss = (current_price - future_low) / current_price
                        
                        # Calculate risk-reward ratio
                        if potential_loss > 0.001:  # Avoid division by very small numbers
                            rr_ratio = potential_gain / potential_loss
                        else:
                            rr_ratio = potential_gain * 100  # High reward if minimal downside
                        
                        # Cap the ratio to reasonable bounds
                        rr_ratio = max(min(rr_ratio, 10.0), 0.01)
                    else:
                        rr_ratio = 0.1
                    
                    future_returns.append(rr_ratio)
                else:
                    # For last few rows, use historical average
                    if len(future_returns) > 0:
                        future_returns.append(np.mean(future_returns[-10:]))
                    else:
                        future_returns.append(1.0)
            
            self.df["Risk_Reward"] = future_returns
            
            # Remove any infinite or extreme values
            self.df["Risk_Reward"] = self.df["Risk_Reward"].replace([np.inf, -np.inf], np.nan)
            self.df["Risk_Reward"] = self.df["Risk_Reward"].fillna(self.df["Risk_Reward"].median())
            
            print(f"✅ Target variables created")
            print(f"📊 Buy Signal distribution: \n{self.df['Buy_Signal'].value_counts()}")
            print(f"📊 Buy Signal percentage: {self.df['Buy_Signal'].mean()*100:.1f}%")
            print(f"📊 Average Risk-Reward: {self.df['Risk_Reward'].mean():.3f}")
            
            # **SAFETY CHECK**: Ensure we have enough buy signals
            buy_signal_pct = self.df['Buy_Signal'].mean() * 100
            if buy_signal_pct < 5:
                print(f"⚠️ Warning: Only {buy_signal_pct:.1f}% buy signals. Adjusting conditions...")
                # If too few signals, make conditions more lenient
                lenient_condition = (
                    (self.df["RSI"] < 60) |  # More lenient RSI
                    (self.df["Momentum"] > self.df["Momentum"].quantile(0.4)) |  # Lower momentum threshold
                    (self.df["close"] > self.df["EMA_10"])  # Simple trend condition
                )
                self.df["Buy_Signal"] = lenient_condition.astype(int)
                print(f"📊 Adjusted Buy Signal percentage: {self.df['Buy_Signal'].mean()*100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating targets: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_training_data(self):
        """Prepare features and targets for training"""
        print("🔄 Preparing training data...")
        
        # Define feature sets
        basic_features = [
            "SMA_10", "EMA_10", "RSI", "Momentum", "Volatility",
            "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal"
        ]
        
        # Check which features are available
        available_features = [f for f in basic_features if f in self.df.columns]
        print(f"📋 Available features: {available_features}")
        
        # **IMPROVED DATA CLEANING**
        print(f"📊 Data before cleaning: {len(self.df)} rows")
        
        # Fill NaN values with forward fill first, then backward fill
        for feature in available_features:
            self.df[feature] = self.df[feature].fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining NaN values with median
        for feature in available_features:
            median_val = self.df[feature].median()
            self.df[feature] = self.df[feature].fillna(median_val)
        
        # For target variables
        self.df["Buy_Signal"] = self.df["Buy_Signal"].fillna(0).astype(int)
        self.df["Risk_Reward"] = self.df["Risk_Reward"].fillna(self.df["Risk_Reward"].median())
        
        # Remove infinite values
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # Final cleanup - only remove rows where ALL features are NaN
        required_cols = available_features + ["Buy_Signal", "Risk_Reward"]
        before_cleaning = len(self.df)
        
        # Instead of dropping all NaN rows, only drop rows where most features are missing
        # Calculate the percentage of non-null values per row
        non_null_pct = self.df[required_cols].notna().sum(axis=1) / len(required_cols)
        self.df = self.df[non_null_pct >= 0.7]  # Keep rows with at least 70% non-null values
        
        # Fill any remaining NaN values
        for col in required_cols:
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(0)
        
        after_cleaning = len(self.df)
        print(f"📊 Data after cleaning: {after_cleaning} rows (removed {before_cleaning - after_cleaning} rows)")
        
        if after_cleaning < 100:
            print(f"❌ Insufficient data after cleaning: {after_cleaning} rows")
            return False, None, None, None
        
        # Prepare features and targets
        X = self.df[available_features]
        y_buy = self.df["Buy_Signal"]
        y_rr = self.df["Risk_Reward"]
        
        print(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Buy signals: {y_buy.sum()} ({y_buy.mean()*100:.1f}%)")
        
        return True, X, y_buy, y_rr
    
    def train_buy_signal_model(self, X, y_buy):
        """Train buy signal classification model"""
        print("🤖 Training buy signal model...")
        
        try:
            # Split data chronologically
            split_idx = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y_buy.iloc[:split_idx], y_buy.iloc[split_idx:]
            
            print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # **SIMPLIFIED HYPERPARAMETERS** for faster training
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "min_samples_leaf": [10, 20]
            }
            
            # Grid search with time series cross-validation
            rf_classifier = RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for speed
            grid_search = GridSearchCV(
                rf_classifier,
                param_grid,
                scoring="roc_auc",
                cv=tscv,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.best_buy_model = grid_search.best_estimator_
            
            # Evaluate model
            y_pred = self.best_buy_model.predict(X_test)
            y_pred_proba = self.best_buy_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"✅ Buy Signal Model Results:")
            print(f"   Best parameters: {grid_search.best_params_}")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   ROC AUC: {roc_auc:.4f}")
            
            return True, X_test, y_test, y_pred, y_pred_proba
            
        except Exception as e:
            print(f"❌ Error training buy signal model: {e}")
            import traceback
            traceback.print_exc()
            return False, None, None, None, None
    
    def train_risk_reward_model(self, X, y_rr):
        """Train risk-reward regression model"""
        print("📈 Training risk-reward model...")
        
        try:
            # Split data chronologically
            split_idx = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y_rr.iloc[:split_idx], y_rr.iloc[split_idx:]
            
            # **SIMPLIFIED HYPERPARAMETERS**
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "min_samples_leaf": [10, 20]
            }
            
            # Grid search
            rf_regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                rf_regressor,
                param_grid,
                scoring="neg_mean_squared_error",
                cv=tscv,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.best_rr_model = grid_search.best_estimator_
            
            # Evaluate model
            y_pred = self.best_rr_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            print(f"✅ Risk-Reward Model Results:")
            print(f"   Best parameters: {grid_search.best_params_}")
            print(f"   MSE: {mse:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   Mean actual RR: {y_test.mean():.3f}")
            print(f"   Mean predicted RR: {y_pred.mean():.3f}")
            
            return True, X_test, y_test, y_pred
            
        except Exception as e:
            print(f"❌ Error training risk-reward model: {e}")
            import traceback
            traceback.print_exc()
            return False, None, None, None
    
    def save_models(self):
        """Save trained models"""
        print("💾 Saving models...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.best_buy_model is not None:
                buy_model_path = os.path.join(self.models_dir, "buy_model_latest.pkl")
                buy_model_backup = os.path.join(self.models_dir, f"buy_model_{timestamp}.pkl")
                
                joblib.dump(self.best_buy_model, buy_model_path)
                joblib.dump(self.best_buy_model, buy_model_backup)
                print(f"✅ Buy model saved to {buy_model_path}")
            
            if self.best_rr_model is not None:
                rr_model_path = os.path.join(self.models_dir, "rr_model_latest.pkl")
                rr_model_backup = os.path.join(self.models_dir, f"rr_model_{timestamp}.pkl")
                
                joblib.dump(self.best_rr_model, rr_model_path)
                joblib.dump(self.best_rr_model, rr_model_backup)
                print(f"✅ Risk-reward model saved to {rr_model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
    
    def generate_training_report(self, buy_results=None, rr_results=None):
        """Generate comprehensive training report"""
        print("📊 Generating training report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.reports_dir, f"training_report_{timestamp}.txt")
            
            with open(report_file, 'w') as f:
                f.write(f"TRADING MODEL TRAINING REPORT\n")
                f.write(f"================================\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Symbol: {self.symbol}\n")
                f.write(f"Data points: {len(self.df)}\n")
                f.write(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}\n\n")
                
                # Data statistics
                f.write(f"DATA STATISTICS:\n")
                f.write(f"- Buy signals: {self.df['Buy_Signal'].sum()} ({self.df['Buy_Signal'].mean()*100:.1f}%)\n")
                f.write(f"- Average risk-reward: {self.df['Risk_Reward'].mean():.3f}\n")
                f.write(f"- Price range: ₹{self.df['close'].min():.2f} - ₹{self.df['close'].max():.2f}\n\n")
                
                if buy_results and len(buy_results) >= 4:
                    X_test, y_test, y_pred, y_pred_proba = buy_results
                    accuracy = accuracy_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    
                    f.write(f"BUY SIGNAL MODEL:\n")
                    f.write(f"- Model type: Random Forest Classifier\n")
                    f.write(f"- Test accuracy: {accuracy:.4f}\n")
                    f.write(f"- ROC AUC: {roc_auc:.4f}\n\n")
                
                if rr_results and len(rr_results) >= 3:
                    X_test, y_test, y_pred = rr_results
                    mse = mean_squared_error(y_test, y_pred)
                    
                    f.write(f"RISK-REWARD MODEL:\n")
                    f.write(f"- Model type: Random Forest Regressor\n")
                    f.write(f"- Test MSE: {mse:.4f}\n\n")
            
            print(f"✅ Training report saved to {report_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return False
    
    def run_complete_training(self, symbol="NIFTY", days_back=360):
        """Run complete model training pipeline"""
        print(f"🚀 Starting complete training pipeline for {symbol}")
        print("=" * 60)
        
        self.symbol = symbol
        
        # Step 1: Load data
        if not self.load_and_prepare_data(days_back):
            return False
        
        # Step 2: Feature engineering
        if not self.feature_engineering():
            return False
        
        # Step 3: Create targets
        if not self.create_targets():
            return False
        
        # Step 4: Prepare training data
        success, X, y_buy, y_rr = self.prepare_training_data()
        if not success:
            return False
        
        # Step 5: Train buy signal model
        buy_success, *buy_results = self.train_buy_signal_model(X, y_buy)
        
        # Step 6: Train risk-reward model
        rr_success, *rr_results = self.train_risk_reward_model(X, y_rr)
        
        # Step 7: Save models
        if buy_success or rr_success:
            self.save_models()
        
        # Step 8: Generate report
        self.generate_training_report(
            buy_results if buy_success else None,
            rr_results if rr_success else None
        )
        
        print("\n" + "=" * 60)
        print("🎉 Training pipeline completed!")
        print(f"✅ Buy signal model: {'Success' if buy_success else 'Failed'}")
        print(f"✅ Risk-reward model: {'Success' if rr_success else 'Failed'}")
        print("=" * 60)
        
        return buy_success and rr_success

if __name__ == "__main__":
    print("🤖 Model Retraining Script")
    print("=" * 50)
    
    # Configuration
    SYMBOL = "NIFTY"
    DAYS_BACK = 360
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(SYMBOL)
        
        # Run training
        success = trainer.run_complete_training(SYMBOL, DAYS_BACK)
        
        if success:
            print(f"\n🎉 Training completed successfully!")
            print(f"📁 Models saved in '{trainer.models_dir}/' directory")
            print(f"📊 Reports saved in '{trainer.reports_dir}/' directory")
            print("\n💡 You can now use the updated models in your trading application!")
        else:
            print(f"\n❌ Training failed. Please check the logs above.")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
