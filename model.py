import pandas as pd
import joblib
import numpy as np
import warnings
from typing import Tuple, Optional
import os

warnings.filterwarnings("ignore")

# === Model Loading ===
def load_models(model_dir="models"):
    """Load trained models with error handling"""
    try:
        buy_model_path = os.path.join(model_dir, "buy_model_latest.pkl")
        rr_model_path = os.path.join(model_dir, "rr_model_latest.pkl")
        
        if not os.path.exists(buy_model_path):
            raise FileNotFoundError(f"Buy model not found at {buy_model_path}")
        if not os.path.exists(rr_model_path):
            raise FileNotFoundError(f"Risk-reward model not found at {rr_model_path}")
        
        buy_model = joblib.load(buy_model_path)
        rr_model = joblib.load(rr_model_path)
        
        print(f"✅ Models loaded successfully from {model_dir}/")
        return buy_model, rr_model
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

# Load models at module level
try:
    buy_model, rr_model = load_models()
except:
    print("⚠️ Models not loaded. Run training script first.")
    buy_model, rr_model = None, None

# === Technical Indicators ===
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI)
    
    Args:
        series: Price series (typically close prices)
        period: Lookback period for RSI calculation
        
    Returns:
        RSI values as pandas Series
    """
    if len(series) < period:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / (loss.replace(0, 1e-8))
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def compute_macd(series: pd.Series, span1: int = 12, span2: int = 26, span_signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence)
    
    Args:
        series: Price series (typically close prices)
        span1: Fast EMA period
        span2: Slow EMA period  
        span_signal: Signal line EMA period
        
    Returns:
        Tuple of (MACD line, Signal line)
    """
    if len(series) < span2:
        nan_series = pd.Series([np.nan] * len(series), index=series.index)
        return nan_series, nan_series
    
    ema1 = series.ewm(span=span1, adjust=False).mean()
    ema2 = series.ewm(span=span2, adjust=False).mean()
    macd = ema1 - ema2
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    
    return macd, signal

def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands
    
    Args:
        series: Price series
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(series) < period:
        nan_series = pd.Series([np.nan] * len(series), index=series.index)
        return nan_series, nan_series, nan_series
    
    middle_band = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR values
    """
    if len(high) < 2:
        return pd.Series([np.nan] * len(high), index=high.index)
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                      k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Stochastic Oscillator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D smoothing period
        
    Returns:
        Tuple of (%K, %D)
    """
    if len(high) < k_period:
        nan_series = pd.Series([np.nan] * len(high), index=high.index)
        return nan_series, nan_series
    
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent

# === Feature Engineering ===
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators to dataframe
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Basic moving averages
    df['SMA_5'] = df['close'].rolling(5).mean()
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    
    # Exponential moving averages
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # Price-based indicators
    df['RSI'] = compute_rsi(df['close'])
    df['MACD'], df['MACD_Signal'] = compute_macd(df['close'])
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = compute_bollinger_bands(df['close'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volatility indicators
    df['ATR'] = compute_atr(df['high'], df['low'], df['close'])
    df['Volatility'] = df['close'].rolling(10).std()
    df['Price_Change'] = df['close'].pct_change()
    
    # Momentum indicators
    df['Momentum'] = df['close'] - df['close'].shift(5)
    df['ROC'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    
    # Stochastic
    df['Stoch_K'], df['Stoch_D'] = compute_stochastic(df['high'], df['low'], df['close'])
    
    # Volume indicators
    df['Volume_SMA'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
    df['Price_Volume'] = df['close'] * df['volume']
    df['VWAP'] = df['Price_Volume'].rolling(20).sum() / df['volume'].rolling(20).sum()
    
    # Lagged features
    df['Lag_Close'] = df['close'].shift(1)
    df['Lag_Volume'] = df['volume'].shift(1)
    df['Lag_RSI'] = df['RSI'].shift(1)
    df['Lag_Momentum'] = df['Momentum'].shift(1)
    
    # Price patterns
    df['Higher_High'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['Lower_Low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['Up_Day'] = (df['close'] > df['open']).astype(int)
    
    return df

def get_model_features() -> list:
    """Get list of features used by the model"""
    return [
        "SMA_10", "EMA_10", "RSI", "Momentum", "Volatility", 
        "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal"
    ]

# === Prediction Functions ===
def predict_buy_signal(df: pd.DataFrame, model=None) -> Tuple[int, float]:
    """
    Predict buy signal using trained model
    
    Args:
        df: DataFrame with features
        model: Trained model (uses global buy_model if None)
        
    Returns:
        Tuple of (signal, confidence)
    """
    if model is None:
        model = buy_model
        
    if model is None:
        raise ValueError("No model available. Please train model first.")
    
    features = get_model_features()
    
    # Get latest features
    latest_features = df[features].iloc[-1:].fillna(0)
    
    # Predict
    prediction_proba = model.predict_proba(latest_features)[0]
    signal = int(prediction_proba[1] > 0.5)
    confidence = prediction_proba[1] * 100
    
    return signal, confidence

def predict_risk_reward(df: pd.DataFrame, model=None) -> float:
    """
    Predict risk-reward ratio using trained model
    
    Args:
        df: DataFrame with features
        model: Trained model (uses global rr_model if None)
        
    Returns:
        Predicted risk-reward ratio
    """
    if model is None:
        model = rr_model
        
    if model is None:
        raise ValueError("No risk-reward model available. Please train model first.")
    
    features = get_model_features()
    
    # Get latest features
    latest_features = df[features].iloc[-1:].fillna(0)
    
    # Predict
    rr_prediction = model.predict(latest_features)[0]
    
    return max(rr_prediction, 0.01)  # Ensure positive value

def generate_signal_with_model(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive signal using both models
    
    Args:
        df: DataFrame with OHLCV and technical indicators
        
    Returns:
        Dictionary with signal information
    """
    if buy_model is None or rr_model is None:
        raise ValueError("Models not loaded. Please run training script first.")
    
    # Ensure we have enough data
    if len(df) < 50:
        return {
            'signal': 'HOLD',
            'confidence': 0,
            'reason': 'Insufficient data for prediction'
        }
    
    # Add technical indicators if not present
    required_features = get_model_features()
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        df = add_technical_indicators(df)
    
    # Generate predictions
    buy_signal, confidence = predict_buy_signal(df)
    risk_reward = predict_risk_reward(df)
    
    # Determine action
    if confidence >= 65:
        action = 'BUY'
    elif confidence <= 35:
        action = 'SELL'
    else:
        action = 'HOLD'
    
    # Get current values
    current_price = df['close'].iloc[-1]
    current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None
    current_macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else None
    
    return {
        'signal': action,
        'confidence': round(confidence, 2),
        'buy_probability': round(confidence/100, 3),
        'risk_reward_ratio': round(risk_reward, 3),
        'current_price': round(current_price, 2),
        'technical_data': {
            'rsi': round(current_rsi, 1) if current_rsi else None,
            'macd': round(current_macd, 4) if current_macd else None,
        },
        'timestamp': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None
    }

# === Model Information ===
def get_model_info():
    """Get information about loaded models"""
    info = {
        'buy_model_loaded': buy_model is not None,
        'rr_model_loaded': rr_model is not None,
        'required_features': get_model_features()
    }
    
    if buy_model is not None:
        info['buy_model_type'] = type(buy_model).__name__
        if hasattr(buy_model, 'feature_importances_'):
            feature_names = get_model_features()
            importances = buy_model.feature_importances_
            info['feature_importance'] = dict(zip(feature_names, importances))
    
    if rr_model is not None:
        info['rr_model_type'] = type(rr_model).__name__
    
    return info

if __name__ == "__main__":
    print("🤖 Model Module Test")
    print("=" * 40)
    
    # Display model info
    model_info = get_model_info()
    print("📊 Model Information:")
    for key, value in model_info.items():
        if key != 'feature_importance':
            print(f"   {key}: {value}")
    
    # Test with sample data
    print("\n🧪 Testing with sample data...")
    
    # Create sample dataframe
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.randn(100).cumsum() * 2,
        'high': 100 + np.random.randn(100).cumsum() * 2 + abs(np.random.randn(100)),
        'low': 100 + np.random.randn(100).cumsum() * 2 - abs(np.random.randn(100)),
        'close': 100 + np.random.randn(100).cumsum() * 2,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    try:
        # Add technical indicators
        sample_df = add_technical_indicators(sample_df)
        print(f"✅ Added technical indicators. DataFrame shape: {sample_df.shape}")
        
        # Generate signal (only if models are loaded)
        if buy_model is not None and rr_model is not None:
            signal_result = generate_signal_with_model(sample_df)
            print(f"✅ Generated signal: {signal_result}")
        else:
            print("⚠️ Skipping signal generation - models not loaded")
            
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
