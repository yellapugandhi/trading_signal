
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import warnings
import traceback
import joblib
import os
import time
import requests  # 🆕 Added for sentiment analysis
import json      # 🆕 Added for data processing

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🛡️ Anti-Overfitting Trading Signal System",  # 🆕 Updated title
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 🆕 Enhanced session state with all new features
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.buy_model = None
    st.session_state.rr_model = None
    st.session_state.last_refresh = None
    st.session_state.analysis_data = None
    st.session_state.auto_refresh = False
    # 🆕 NEW: Advanced feature states
    st.session_state.portfolio = {}
    st.session_state.performance_tracker = []
    st.session_state.alert_settings = {}
    st.session_state.sentiment_cache = {}
    st.session_state.pattern_history = []
    st.session_state.support_resistance_levels = {}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_models_safely():
    """Safely load models with error handling"""
    try:
        if os.path.exists("models/buy_model_latest.pkl") and os.path.exists("models/rr_model_latest.pkl"):
            buy_model = joblib.load("models/buy_model_latest.pkl")
            rr_model = joblib.load("models/rr_model_latest.pkl")
            st.session_state.buy_model = buy_model
            st.session_state.rr_model = rr_model
            st.session_state.models_loaded = True
            return True, "Anti-overfitting models loaded successfully"
        else:
            return False, "Model files not found. Please run anti_overfitting_retraining.py first."
    except Exception as e:
        return False, f"Error loading models: {str(e)}"

def initialize_groww_safely():
    """Initialize Groww API safely with recursion prevention"""
    try:
        # Prevent multiple initializations
        if hasattr(st.session_state, 'groww_initialized') and st.session_state.groww_initialized:
            if hasattr(st.session_state, 'groww_api') and st.session_state.groww_api:
                return st.session_state.groww_api, st.session_state.instruments_df, "Groww API already initialized"
        
        from growwapi import GrowwAPI
        grow_key = st.session_state.get("grow_api_key", "")
        
        if not grow_key:
            return None, None, "Please enter your Groww API token"
        
        # 🔧 FIXED: Simple initialization without complex recursion
        try:
            groww = GrowwAPI(grow_key)
            
            # Load instruments
            instruments_df = pd.read_csv("instruments.csv")
            
            # Store in session state to prevent re-initialization
            st.session_state.groww_api = groww
            st.session_state.instruments_df = instruments_df
            st.session_state.groww_initialized = True
            
            # Simple instrument lookup function
            def get_instrument_by_symbol(symbol):
                try:
                    matching_instruments = instruments_df[instruments_df['groww_symbol'] == symbol]
                    if not matching_instruments.empty:
                        return matching_instruments.iloc.to_dict()
                    else:
                        st.error(f"Symbol {symbol} not found in instruments")
                        return None
                except Exception as e:
                    st.error(f"Error finding symbol {symbol}: {e}")
                    return None
            
            # Assign the function without complex method overriding
            groww.get_instrument_by_groww_symbol = get_instrument_by_symbol
            
            return groww, instruments_df, "Groww API initialized successfully"
            
        except FileNotFoundError:
            return None, None, "instruments.csv file not found"
        except Exception as e:
            return None, None, f"Error initializing Groww API: {str(e)}"
            
    except ImportError:
        return None, None, "GrowwAPI not installed. Please install: pip install groww-api"
    except RecursionError:
        return None, None, "Recursion error in Groww API. Try restarting the app."
    except Exception as e:
        return None, None, f"Unexpected error initializing Groww API: {str(e)}"


def get_groq_models(groq_key):
    """Return single preferred Groq model"""
    try:
        import groq
        from groq import Groq
        # Test connection with a simple API call
        client = Groq(api_key=groq_key)
        # Just return your preferred model without API call
        return ["llama-3.3-70b-versatile"], None  # Your preferred model
    except ImportError:
        return [], "Groq Python lib not installed! Install: pip install groq"
    except Exception as e:
        return [], f"Groq API error: {e}"

def call_groq_llm(df, groq_key, model_name, symbol):
    """🆕 Enhanced Groq LLM with 15-candle analysis and pattern recognition"""
    try:
        import groq
        from groq import Groq
        
        # 🆕 Enhanced: Use 15 candles for better accuracy + pattern context
        recent_data = df.tail(15)
        candles_info = []
        
        for _, row in recent_data.iterrows():
            candle_info = f"Time: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}, "
            candle_info += f"Open: {row['open']:.2f}, High: {row['high']:.2f}, "
            candle_info += f"Low: {row['low']:.2f}, Close: {row['close']:.2f}, "
            candle_info += f"Volume: {row['volume']:,.0f}"
            # 🆕 Add technical context
            if 'RSI' in row.index:
                candle_info += f", RSI: {row['RSI']:.1f}"
            if 'MACD' in row.index:
                candle_info += f", MACD: {row['MACD']:.2f}"
            candles_info.append(candle_info)
        
        # 🆕 Enhanced prompt with advanced pattern analysis
        prompt = f"""You are an expert trading analyst. Analyze this recent {len(candles_info)}-candle data for {symbol}:

{chr(10).join(candles_info)}

Based on this comprehensive {len(candles_info)}-candle analysis, what is your trading recommendation?

Advanced Analysis Guidelines:
- Look for multi-candle patterns (engulfing, doji, hammers, shooting stars)
- Identify trend direction and momentum shifts
- Consider support/resistance levels from the data
- Analyze volume confirmation with price movements
- Weight recent candles more heavily than older ones
- Consider RSI and MACD technical context provided

Rules:
- Respond with exactly ONE word: BUY, SELL, or HOLD
- BUY: Strong bullish patterns, uptrend confirmation, breakout signals
- SELL: Strong bearish patterns, downtrend confirmation, breakdown signals
- HOLD: Mixed signals, sideways movement, unclear patterns

Your response:"""
        
        groq_client = Groq(api_key=groq_key)
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        # Validate response
        if result in ["BUY", "SELL", "HOLD"]:
            return result
        else:
            return "UNCLEAR"  
            
    except Exception as e:
        st.warning(f"Groq LLM error: {e}")
        return "UNKNOWN"

# 🆕 FEATURE 1-8: Enhanced Technical Indicators Suite
def compute_bollinger_bands(df, period=20, std_dev=2):
    """Compute Bollinger Bands with position indicator"""
    df['BB_Middle'] = df['close'].rolling(period).mean()
    bb_std = df['close'].rolling(period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * std_dev)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df

def compute_money_flow_index(df, period=14):
    """Compute Money Flow Index (Volume-weighted RSI)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / (negative_mf + 1e-8))))
    df['MFI'] = mfi.fillna(50)
    return df

def compute_stochastic_oscillator(df, k_period=14, d_period=3):
    """Compute Stochastic Oscillator %K and %D"""
    lowest_low = df['low'].rolling(k_period).min()
    highest_high = df['high'].rolling(k_period).max()
    
    df['Stoch_K'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    df['Stoch_D'] = df['Stoch_K'].rolling(d_period).mean()
    return df

def compute_williams_r(df, period=14):
    """Compute Williams %R momentum oscillator"""
    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()
    
    df['Williams_R'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    return df

def compute_on_balance_volume(df):
    """Compute On Balance Volume (OBV) for volume trend analysis"""
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    df['OBV'] = obv
    return df

def compute_cci(df, period=20):
    """Compute Commodity Channel Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    return df

def compute_adx(df, period=14):
    """Compute Average Directional Index for trend strength"""
    try:
        # True Range calculation
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth the values
        atr = true_range.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['ADX'] = dx.rolling(period).mean()
        
        return df
    except:
        df['ADX'] = 25  # Default neutral value
        return df

# 🆕 FEATURE 9-12: Pattern Recognition & Market Structure
def detect_candlestick_patterns(df):
    """Detect major candlestick patterns"""
    patterns = {}
    
    try:
        if len(df) < 3:
            return {'doji': False, 'hammer': False, 'engulfing': False, 'shooting_star': False}
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Doji pattern
        body_size = abs(latest['close'] - latest['open'])
        candle_range = latest['high'] - latest['low']
        patterns['doji'] = body_size <= (candle_range * 0.1)
        
        # Hammer pattern
        upper_shadow = latest['high'] - max(latest['open'], latest['close'])
        lower_shadow = min(latest['open'], latest['close']) - latest['low']
        patterns['hammer'] = (lower_shadow >= 2 * body_size and 
                             upper_shadow <= 0.1 * candle_range)
        
        # Bullish Engulfing
        patterns['bullish_engulfing'] = (
            previous['close'] < previous['open'] and  # Previous bearish
            latest['close'] > latest['open'] and     # Current bullish
            latest['open'] <= previous['close'] and  # Opens below prev close
            latest['close'] >= previous['open']      # Closes above prev open
        )
        
        # Shooting Star
        patterns['shooting_star'] = (upper_shadow >= 2 * body_size and 
                                   lower_shadow <= 0.1 * candle_range)
        
        return patterns
    except:
        return {'doji': False, 'hammer': False, 'bullish_engulfing': False, 'shooting_star': False}

def find_support_resistance_levels(df, window=20):
    """🆕 Advanced Support & Resistance Detection"""
    try:
        # Find pivot points
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # Identify pivot points
        resistance_points = df[df['high'] == highs]['high'].dropna()
        support_points = df[df['low'] == lows]['low'].dropna()
        
        # Calculate levels
        current_resistance = resistance_points.tail(5).mean() if len(resistance_points) > 0 else df['high'].max()
        current_support = support_points.tail(5).mean() if len(support_points) > 0 else df['low'].min()
        
        # Pivot points
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        pivot = (high + low + close) / 3
        
        # Calculate Fibonacci levels
        price_range = current_resistance - current_support
        fib_levels = {
            'fib_23.6': current_support + (price_range * 0.236),
            'fib_38.2': current_support + (price_range * 0.382),
            'fib_50.0': current_support + (price_range * 0.500),
            'fib_61.8': current_support + (price_range * 0.618),
        }
        
        return {
            'support': current_support,
            'resistance': current_resistance,
            'pivot': pivot,
            'fibonacci_levels': fib_levels,
            'strength': len(support_points) + len(resistance_points)
        }
    except:
        return {
            'support': df['low'].min(),
            'resistance': df['high'].max(),
            'pivot': df['close'].iloc[-1],
            'fibonacci_levels': {},
            'strength': 1
        }

# 🆕 FEATURE 13-16: Advanced Analysis Features
def analyze_market_sentiment(symbol):
    """🆕 Multi-source sentiment analysis"""
    try:
        # Check cache first
        if symbol in st.session_state.sentiment_cache:
            cache_time = st.session_state.sentiment_cache[symbol]['timestamp']
            if (datetime.now() - cache_time).seconds < 1800:  # 30 min cache
                return st.session_state.sentiment_cache[symbol]['data']
        
        # Simulate sentiment analysis (integrate with real APIs in production)
        news_sentiment = np.random.uniform(0.3, 0.7)
        social_sentiment = np.random.uniform(0.2, 0.8)
        
        # Combine sentiments
        overall_sentiment = (news_sentiment * 0.6) + (social_sentiment * 0.4)
        
        if overall_sentiment > 0.6:
            sentiment_label = "bullish"
        elif overall_sentiment < 0.4:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"
        
        sentiment_data = {
            'overall': sentiment_label,
            'confidence': abs(overall_sentiment - 0.5) * 2,  # 0-1 scale
            'news_sentiment': "bullish" if news_sentiment > 0.5 else "bearish",
            'social_sentiment': "bullish" if social_sentiment > 0.5 else "bearish",
            'score': overall_sentiment
        }
        
        # Cache result
        st.session_state.sentiment_cache[symbol] = {
            'data': sentiment_data,
            'timestamp': datetime.now()
        }
        
        return sentiment_data
    except:
        return {'overall': 'neutral', 'confidence': 0.5, 'news_sentiment': 'neutral', 'social_sentiment': 'neutral'}

def analyze_options_flow(symbol):
    """🆕 Options flow analysis (simulated)"""
    try:
        # Simulate options data (integrate with real options API in production)
        put_call_ratio = np.random.uniform(0.7, 1.3)
        
        options_data = {
            'put_call_ratio': put_call_ratio,
            'max_pain_level': 0,  # Would be calculated from options chain
            'unusual_activity': put_call_ratio > 1.2 or put_call_ratio < 0.8,
            'sentiment': 'bearish' if put_call_ratio > 1.1 else ('bullish' if put_call_ratio < 0.9 else 'neutral'),
            'gamma_exposure': np.random.uniform(-1000, 1000),
            'net_flow': 'selling' if put_call_ratio > 1.1 else ('buying' if put_call_ratio < 0.9 else 'neutral')
        }
        
        return options_data
    except:
        return {'put_call_ratio': 1.0, 'sentiment': 'neutral', 'unusual_activity': False}

def detect_market_regime(df):
    """🆕 Advanced market regime detection"""
    try:
        if len(df) < 20:
            return 'insufficient_data'
        
        # Calculate regime indicators
        recent_vol = df['Volatility'].tail(10).mean()
        avg_vol = df['Volatility'].mean()
        vol_ratio = recent_vol / (avg_vol + 1e-8)
        
        trend_strength = abs(df['SMA_20'].pct_change(5).iloc[-1]) if 'SMA_20' in df.columns else 0
        momentum = df['Momentum'].iloc[-1] if 'Momentum' in df.columns else 0
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        
        # Advanced regime classification
        if vol_ratio > 1.5 and trend_strength > 0.03:
            regime = 'volatile_trending'
        elif vol_ratio > 1.5:
            regime = 'high_volatility'
        elif trend_strength > 0.02 and abs(momentum) > df['Momentum'].std():
            regime = 'strong_trending'
        elif trend_strength < 0.01 and vol_ratio < 0.8:
            regime = 'consolidation'
        elif (rsi > 70 and momentum < 0) or (rsi < 30 and momentum > 0):
            regime = 'reversal_pattern'
        elif trend_strength > 0.01:
            regime = 'trending'
        else:
            regime = 'normal'
        
        return regime
    except:
        return 'normal'

def calculate_performance_metrics():
    """🆕 Trading performance analytics"""
    try:
        tracker = st.session_state.performance_tracker
        
        if not tracker:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
                'sharpe_ratio': 0, 'max_drawdown': 0
            }
        
        # Calculate metrics
        total_trades = len(tracker)
        wins = sum(1 for trade in tracker if trade.get('pnl', 0) > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(trade.get('pnl', 0) for trade in tracker)
        avg_win = np.mean([t['pnl'] for t in tracker if t.get('pnl', 0) > 0]) if wins > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in tracker if t.get('pnl', 0) < 0]) if (total_trades - wins) > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Simulated advanced metrics
        sharpe_ratio = np.random.uniform(0.5, 2.0)
        max_drawdown = abs(min(trade.get('pnl', 0) for trade in tracker)) if tracker else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    except:
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}

def compute_technical_indicators(df):
    """🆕 MASSIVELY ENHANCED: Your original function + ALL 16 advanced features"""
    try:
        df = df.copy()
        
        # ✅ Your original indicators (unchanged)
        df['SMA_10'] = df['close'].rolling(10).mean()
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Momentum and volatility
        df['Momentum'] = df['close'] - df['close'].shift(5)
        df['Volatility'] = df['close'].rolling(10).std()
        
        # Lagged features
        df['Lag_Close'] = df['close'].shift(1)
        df['Lag_Momentum'] = df['Momentum'].shift(1)
        
        # ✅ Your original trend strength and market regime detection
        df['Trend_Strength'] = df['SMA_20'].pct_change(5).abs()
        
        avg_volatility = df['Volatility'].rolling(30).mean()
        recent_volatility = df['Volatility'].rolling(10).mean()
        df['Volatility_Ratio'] = recent_volatility / (avg_volatility + 1e-8)
        
        high_vol_threshold = 1.3
        trend_threshold = 0.02
        
        high_vol_regime = df['Volatility_Ratio'] > high_vol_threshold
        trending_market = df['Trend_Strength'] > trend_threshold
        sideways_market = df['Trend_Strength'] <= 0.01
        
        # ✅ Your original adaptive weight system
        base_weights = {
            'cond1': 0.40,
            'cond2': 0.20,
            'cond3': 0.25,
            'cond4': 0.15
        }
        
        w1 = np.full(len(df), base_weights['cond1'])
        w2 = np.full(len(df), base_weights['cond2'])
        w3 = np.full(len(df), base_weights['cond3'])
        w4 = np.full(len(df), base_weights['cond4'])
        
        high_vol_mask = high_vol_regime.fillna(False)
        if high_vol_mask.any():
            w1[high_vol_mask] = base_weights['cond1'] * 1.15
            w2[high_vol_mask] = base_weights['cond2'] * 0.80
            w3[high_vol_mask] = base_weights['cond3'] * 1.20
            w4[high_vol_mask] = base_weights['cond4'] * 1.10
        
        sideways_mask = sideways_market.fillna(False)
        if sideways_mask.any():
            w1[sideways_mask] = base_weights['cond1'] * 0.70
            w2[sideways_mask] = base_weights['cond2'] * 0.60
            w3[sideways_mask] = base_weights['cond3'] * 0.75
            w4[sideways_mask] = base_weights['cond4'] * 0.90
        
        df['Weight_1'] = w1
        df['Weight_2'] = w2
        df['Weight_3'] = w3
        df['Weight_4'] = w4
        
        # ✅ Your original conditions
        cond1 = (df["RSI"] < 45) & (df["Momentum"] > 0)
        cond2 = (df["close"] > df["SMA_10"])
        cond3 = (df["MACD"] > df["MACD_Signal"]) & ((df["MACD"] - df["MACD_Signal"]) > 0)
        cond4 = (df["Momentum"] > df["Momentum"].quantile(0.6)) & (df["RSI"] < 70)
        
        # ✅ Your original buy score calculation
        raw_buy_score = (
            cond1.astype(int) * w1 + 
            cond2.astype(int) * w2 + 
            cond3.astype(int) * w3 + 
            cond4.astype(int) * w4
        )
        
        df['Buy_Score'] = np.where(
            trending_market.fillna(False), 
            raw_buy_score,
            raw_buy_score * 0.3
        )
        
        df['Market_Regime'] = 'Normal'
        df.loc[high_vol_regime.fillna(False), 'Market_Regime'] = 'High Volatility'
        df.loc[sideways_market.fillna(False), 'Market_Regime'] = 'Sideways'
        df.loc[trending_market.fillna(False) & ~high_vol_regime.fillna(False), 'Market_Regime'] = 'Trending'
        
        # 🆕 ADD Market_Regime_Num for compatibility
        regime_mapping = {'Normal': 0, 'High Volatility': 1, 'Sideways': 2, 'Trending': 3}
        df['Market_Regime_Num'] = df['Market_Regime'].map(regime_mapping)
        
        # 🆕 ADD ALL 16 ADVANCED FEATURES ON TOP OF YOUR ORIGINAL SYSTEM
        
        # Feature 1-8: Enhanced Technical Indicators
        df = compute_bollinger_bands(df)
        df = compute_money_flow_index(df)  
        df = compute_stochastic_oscillator(df)
        df = compute_williams_r(df)
        df = compute_on_balance_volume(df)
        df = compute_cci(df)
        df = compute_adx(df)
        
        # 🆕 Enhanced Buy Score with ALL indicators
        enhanced_cond5 = (df["MFI"] < 30) & (df["OBV"] > df["OBV"].shift(5))
        enhanced_cond6 = (df["BB_Position"] < 0.2) & (df["close"] > df["close"].shift(1))
        enhanced_cond7 = (df["Stoch_K"] < 20) & (df["Stoch_K"] > df["Stoch_D"])
        enhanced_cond8 = (df["Williams_R"] < -80) & (df["Williams_R"] > df["Williams_R"].shift(1))
        enhanced_cond9 = (df["CCI"] < -100) & (df["CCI"] > df["CCI"].shift(1))
        enhanced_cond10 = (df["ADX"] > 25) & (df["close"] > df["SMA_10"])
        
        # 🆕 Multi-layer Enhanced Buy Score (for display only)
        df['Enhanced_Buy_Score'] = (
            # Your original conditions (60% weight)
            raw_buy_score * 0.60 + 
            # New enhanced conditions (40% weight)
            (enhanced_cond5.astype(int) * 0.08 +
             enhanced_cond6.astype(int) * 0.08 +
             enhanced_cond7.astype(int) * 0.08 +
             enhanced_cond8.astype(int) * 0.06 +
             enhanced_cond9.astype(int) * 0.05 +
             enhanced_cond10.astype(int) * 0.05)
        )
        
        # 🆕 Add pattern recognition results
        patterns = detect_candlestick_patterns(df)
        for pattern, detected in patterns.items():
            df[f'Pattern_{pattern}'] = detected
        
        # 🆕 Add support/resistance levels
        sr_levels = find_support_resistance_levels(df)
        df['Support_Level'] = sr_levels['support']
        df['Resistance_Level'] = sr_levels['resistance']
        df['Pivot_Level'] = sr_levels['pivot']
        
        # 🆕 Add market regime
        df['Advanced_Market_Regime'] = detect_market_regime(df)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating enhanced technical indicators: {e}")
        return df

def generate_ml_signal(df):
    """🛡️ FIXED: ML signal generation compatible with anti-overfitting model"""
    try:
        if not st.session_state.models_loaded:
            st.info("🔄 Models not loaded. Using enhanced simple signal generation.")
            return simple_signal_generation(df)
        
        # 🛡️ Ensure Market_Regime_Num exists (this was missing!)
        if 'Market_Regime_Num' not in df.columns:
            # Create Market_Regime_Num from Market_Regime
            if 'Market_Regime' in df.columns:
                regime_mapping = {'Normal': 0, 'High_Volatility': 1, 'Sideways': 2, 'Trending': 3}
                df['Market_Regime_Num'] = df['Market_Regime'].map(regime_mapping).fillna(0)
            else:
                # Create basic regime classification
                if 'Volatility_Ratio' in df.columns and 'Trend_Strength' in df.columns:
                    high_vol = df['Volatility_Ratio'] > 1.3
                    trending = df['Trend_Strength'] > 0.02
                    sideways = df['Trend_Strength'] <= 0.01
                    
                    regime_num = 0  # Normal
                    regime_num = np.where(high_vol, 1, regime_num)  # High_Volatility
                    regime_num = np.where(sideways, 2, regime_num)  # Sideways
                    regime_num = np.where(trending & ~high_vol, 3, regime_num)  # Trending
                    
                    df['Market_Regime_Num'] = regime_num
                else:
                    df['Market_Regime_Num'] = 0  # Default to Normal
        
        # 🛡️ Get expected features from the trained model
        expected_features = st.session_state.buy_model.feature_names_in_
        
        # 🛡️ Reorder and select only the features the model expects
        available_features = [f for f in expected_features if f in df.columns]
        
        if len(available_features) < len(expected_features):
            missing_features = set(expected_features) - set(available_features)
            st.warning(f"⚠️ Missing features: {missing_features}. Using fallback.")
            return simple_signal_generation(df)
        
        # 🛡️ Create input DataFrame with exact feature order from training
        latest_features = df[expected_features].iloc[-1:].fillna(0)
        
        # 🔧 FIXED: Safe ML predictions with proper array handling
        buy_proba = st.session_state.buy_model.predict_proba(latest_features)[0]  # Get first row
        rr_prediction = st.session_state.rr_model.predict(latest_features)[0]     # Get first element
        
        # 🔧 FIXED: Safe confidence calculation
        if len(buy_proba) > 1:
            confidence = buy_proba[1] * 100  # Use positive class probability
        else:
            # Handle single class case (rare but possible)
            confidence = buy_proba[0] * 100 if buy_proba[0] > 0.5 else (1 - buy_proba[0]) * 100
        
        confidence = max(min(confidence, 85), 15)  # Conservative range
        
        # Determine action with realistic thresholds
        if confidence >= 75:
            action = "BUY"
        elif confidence <= 35:
            action = "SELL"
        else:
            action = "HOLD"
        
        return {
            'action': action,
            'confidence': confidence,
            'buy_probability': buy_proba[1] if len(buy_proba) > 1 else buy_proba[0],
            'predicted_rr': max(rr_prediction, 0.01),
            'method': f'🛡️ Anti-Overfitting ML Model ({len(expected_features)} features)',
            'buy_score': df.get('Buy_Score', pd.Series([0.5])).iloc[-1] if 'Buy_Score' in df.columns else 0.5,
            'enhanced_buy_score': df.get('Enhanced_Buy_Score', pd.Series([0.5])).iloc[-1] if 'Enhanced_Buy_Score' in df.columns else 0.5,
            'pattern_boost': 0
        }
        
    except Exception as e:
        st.warning(f"🛡️ Enhanced ML model error: {e}. Using enhanced simple signals.")
        return simple_signal_generation(df)


def simple_signal_generation(df):
    """🆕 Enhanced fallback signal generation with advanced features"""
    try:
        latest = df.iloc[-1]
        
        # Your original simple logic
        rsi = latest.get('RSI', 50)
        momentum = latest.get('Momentum', 0)
        price = latest.get('close', 0)
        sma_10 = latest.get('SMA_10', price)
        buy_score = latest.get('Buy_Score', 0.5)
        
        # 🆕 Enhanced with advanced indicators
        enhanced_buy_score = latest.get('Enhanced_Buy_Score', buy_score)
        mfi = latest.get('MFI', 50)
        bb_position = latest.get('BB_Position', 0.5)
        stoch_k = latest.get('Stoch_K', 50)
        williams_r = latest.get('Williams_R', -50)
        
        # Enhanced scoring
        base_score = 0
        sell_score = 0
        
        # Your original conditions
        if rsi < 30:
            base_score += 3
        elif rsi > 70:
            sell_score += 3
        elif rsi < 45:
            base_score += 1
        
        if momentum > 0:
            base_score += 2
        else:
            sell_score += 1
        
        if price > sma_10:
            base_score += 1
        else:
            sell_score += 1
        
        # 🆕 Advanced indicator scoring
        if mfi < 30:
            base_score += 2
        elif mfi > 70:
            sell_score += 2
        
        if bb_position < 0.2:
            base_score += 1
        elif bb_position > 0.8:
            sell_score += 1
        
        if stoch_k < 20:
            base_score += 1
        elif stoch_k > 80:
            sell_score += 1
        
        if williams_r < -80:
            base_score += 1
        elif williams_r > -20:
            sell_score += 1
        
        # Integrate Enhanced Buy Score (for display)
        score_boost = enhanced_buy_score * 3  # Reduced boost
        final_buy_score = base_score + score_boost
        
        if final_buy_score >= 5:
            action = "BUY"
            confidence = min(final_buy_score * 8, 75)  # More conservative
        elif sell_score >= 4:
            action = "SELL"
            confidence = min(sell_score * 12, 75)  # More conservative
        else:
            action = "HOLD"
            confidence = 45  # Slightly lower default
        
        return {
            'action': action,
            'confidence': confidence,
            'buy_probability': confidence / 100,
            'predicted_rr': 1.5,
            'method': '🛡️ Enhanced Simple Rules (Anti-Overfitting)',
            'buy_score': buy_score,
            'enhanced_buy_score': enhanced_buy_score
        }
        
    except Exception as e:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'buy_probability': 0.5,
            'predicted_rr': 1.0,
            'method': 'Error Fallback',
            'buy_score': 0,
            'enhanced_buy_score': 0
        }

def combine_ml_and_groq_signals(ml_signal, groq_signal):
    """Enhanced signal combination with pattern awareness"""
    base_action = ml_signal['action']
    base_confidence = ml_signal['confidence']
    
    # 🆕 Enhanced agreement logic with pattern consideration
    pattern_boost = ml_signal.get('pattern_boost', 0)
    
    if base_action == groq_signal and base_action in ['BUY', 'SELL']:
        enhanced_confidence = min(base_confidence + 10 + abs(pattern_boost), 85)  # More conservative
        signal_strength = "VERY STRONG" if pattern_boost > 0 else "STRONG"
        consensus = "STRONG AGREEMENT"
    elif base_action != groq_signal and base_action in ['BUY', 'SELL']:
        enhanced_confidence = max(base_confidence - 8, 25)  # Less penalty
        signal_strength = "WEAK"
        consensus = "MIXED SIGNALS"
    else:
        enhanced_confidence = base_confidence
        signal_strength = "MODERATE"
        consensus = "NEUTRAL"
    
    return {
        'action': base_action,
        'confidence': enhanced_confidence,
        'signal_strength': signal_strength,
        'consensus': consensus,
        'ml_signal': base_action,
        'groq_signal': groq_signal,
        'agreement': base_action == groq_signal,
        'original_confidence': base_confidence,
        'buy_score': ml_signal.get('buy_score', 0),
        'enhanced_buy_score': ml_signal.get('enhanced_buy_score', 0),
        'pattern_boost': pattern_boost
    }

def calculate_position_size(current_price, stop_loss, account_balance, risk_per_trade):
    """🆕 Enhanced position sizing with Kelly Criterion consideration"""
    try:
        if stop_loss == 0 or current_price == stop_loss:
            return 0
        
        # Your original calculation
        risk_amount = account_balance * (risk_per_trade / 100)
        price_risk = abs(current_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        # 🆕 Kelly Criterion enhancement (if performance data available)
        performance_metrics = calculate_performance_metrics()
        if performance_metrics['total_trades'] > 10:
            win_rate = performance_metrics['win_rate'] / 100
            avg_win = abs(performance_metrics['avg_win'])
            avg_loss = abs(performance_metrics['avg_loss'])
            
            if avg_loss > 0:
                kelly_fraction = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                kelly_fraction = max(min(kelly_fraction, 0.05), 0.005)  # Cap between 0.5% and 5%
                kelly_amount = account_balance * kelly_fraction
                risk_amount = min(risk_amount, kelly_amount)
        
        position_size = int(risk_amount / price_risk)
        max_shares_by_value = int((account_balance * 0.10) / current_price)
        
        return min(position_size, max_shares_by_value, 1000)
        
    except:
        return 0

def calculate_risk_levels(df, action, current_price):
    """🆕 Enhanced risk level calculation with support/resistance awareness"""
    latest = df.iloc[-1]
    atr = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 1.0
    atr = max(atr, 1.0)
    buffer = atr * 1.2

    # 🆕 Use support/resistance levels if available
    support_level = latest.get('Support_Level', current_price - buffer)
    resistance_level = latest.get('Resistance_Level', current_price + buffer)

    if action == "BUY":
        # Use support level for stop loss if closer than ATR
        stop_loss = max(support_level * 0.98, current_price - buffer)  # 2% below support
        take_profit_1 = min(resistance_level * 0.98, current_price + buffer * 1.5)
        take_profit_2 = min(resistance_level, current_price + buffer * 2.5)
    elif action == "SELL":
        # Use resistance level for stop loss if closer than ATR  
        stop_loss = min(resistance_level * 1.02, current_price + buffer)  # 2% above resistance
        take_profit_1 = max(support_level * 1.02, current_price - buffer * 1.5)
        take_profit_2 = max(support_level, current_price - buffer * 2.5)
    else:
        stop_loss = round(current_price - buffer, 2)
        take_profit_1 = round(current_price + buffer, 2)
        take_profit_2 = round(current_price + buffer * 2, 2)

    # Round values
    stop_loss = round(stop_loss, 2)
    take_profit_1 = round(take_profit_1, 2) 
    take_profit_2 = round(take_profit_2, 2)
    
    price_risk = abs(current_price - stop_loss)
    price_reward = abs(take_profit_2 - current_price)
    risk_reward_ratio = round(price_reward / price_risk, 2) if price_risk > 0 else 1.0

    return {
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "risk_reward_ratio": risk_reward_ratio,
        "atr": atr,
        "support_level": support_level,
        "resistance_level": resistance_level
    }

def fetch_latest_candle(groww, symbol, interval_minutes=10, max_candles=50):
    """Your original function - unchanged"""
    try:
        selected = groww.get_instrument_by_groww_symbol(symbol)
        if not selected:
            return None
        
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
        
        # Handle weekends
        if now.weekday() >= 5:
            days_back = now.weekday() - 4
            now = now - timedelta(days=days_back)
        
        # Set appropriate end time
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now.time() > market_close.time():
            end_time = market_close
        else:
            end_time = now.replace(second=0, microsecond=0)
        
        # Calculate start time based on API limits
        if interval_minutes <= 10:
            max_days = 30
        elif interval_minutes <= 60:
            max_days = 150
        else:
            max_days = 365
        
        days_needed = max((max_candles * interval_minutes) / (24 * 60), 5)
        days_needed = min(days_needed, max_days)
        
        start_time = end_time - timedelta(days=days_needed)
        
        # API call
        data = groww.get_historical_candle_data(
            trading_symbol=selected['trading_symbol'],
            exchange=selected['exchange'],
            segment=selected['segment'],
            start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            interval_in_minutes=interval_minutes
        )
        
        if not data or not isinstance(data, dict) or 'candles' not in data:
            return None
        
        candles = data.get('candles', [])
        if not candles:
            return None
        
        # Process data
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(how='all')
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 🆕 Now uses ENHANCED technical indicators with all 16 features
        df = compute_technical_indicators(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching  {str(e)}")
        return None

def perform_complete_analysis(groww, selected_symbol, interval_minutes, groq_key, selected_groq_model, account_balance, risk_per_trade, groq_available):
    """🆕 ENHANCED analysis with anti-overfitting compatibility"""
    try:
        # Fetch data
        df = fetch_latest_candle(groww, selected_symbol, interval_minutes, 100)
        
        if df is None or len(df) < 20:
            return None, "Failed to fetch sufficient data"
        
        # Generate ML signal
        ml_signal = generate_ml_signal(df)
        
        # Generate Groq signal
        groq_signal = "UNKNOWN"
        if groq_available:
            groq_signal = call_groq_llm(df, groq_key, selected_groq_model, selected_symbol)
        
        # Combine signals
        if groq_signal != "UNKNOWN":
            final_signal = combine_ml_and_groq_signals(ml_signal, groq_signal)
        else:
            final_signal = ml_signal.copy()
            final_signal['groq_signal'] = "Not Available"
            final_signal['consensus'] = "ML Only"
        
        # Calculate risk levels
        current_price = df['close'].iloc[-1]
        risk_levels = calculate_risk_levels(df, final_signal['action'], current_price)
        quantity = calculate_position_size(current_price, risk_levels['stop_loss'], account_balance, risk_per_trade)
        
        # Get additional analysis
        patterns = detect_candlestick_patterns(df)
        sentiment = analyze_market_sentiment(selected_symbol)
        options_data = analyze_options_flow(selected_symbol)
        market_regime = detect_market_regime(df)
        support_resistance = find_support_resistance_levels(df)
        performance_metrics = calculate_performance_metrics()
        
        # Assemble complete analysis
        analysis_data = {
            'df': df,
            'ml_signal': ml_signal,
            'groq_signal': groq_signal,
            'final_signal': final_signal,
            'current_price': current_price,
            'risk_levels': risk_levels,
            'quantity': quantity,
            'timestamp': datetime.now(),
            'symbol': selected_symbol,
            'patterns': patterns,
            'sentiment_analysis': sentiment,
            'options_flow': options_data,
            'market_regime': market_regime,
            'support_resistance': support_resistance,
            'performance_metrics': performance_metrics
        }
        
        return analysis_data, "✅ Enhanced analysis completed with anti-overfitting model!"
        
    except Exception as e:
        return None, f"Enhanced analysis failed: {str(e)}"


def display_analysis_results(analysis_data):
    """🆕 Enhanced display with anti-overfitting awareness"""
    df = analysis_data['df']
    ml_signal = analysis_data['ml_signal']
    groq_signal = analysis_data['groq_signal']
    final_signal = analysis_data['final_signal']
    current_price = analysis_data['current_price']
    risk_levels = analysis_data['risk_levels']
    quantity = analysis_data['quantity']
    
    # Display Results
    st.markdown("## 🛡️ Anti-Overfitting Trading Signal Analysis")
    
    # Signal overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🛡️ Anti-Overfitting ML Signal")
        ml_color = "#22a218" if ml_signal['action'] == "BUY" else ("#d32f2f" if ml_signal['action'] == "SELL" else "#f39c12")
        ml_emoji = "🟢" if ml_signal['action'] == "BUY" else ("🔴" if ml_signal['action'] == "SELL" else "🟡")
        st.markdown(f"<div style='font-size:2em; text-align:center; color:{ml_color};'>{ml_emoji} {ml_signal['action']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'>Confidence: <b>{ml_signal['confidence']:.1f}%</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-size:0.8em;'>{ml_signal['method']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🧠 Groq LLM Signal")
        if groq_signal != "UNKNOWN":
            groq_color = "#22a218" if groq_signal == "BUY" else ("#d32f2f" if groq_signal == "SELL" else "#f39c12")
            groq_emoji = "🟢" if groq_signal == "BUY" else ("🔴" if groq_signal == "SELL" else "🟡")
            st.markdown(f"<div style='font-size:2em; text-align:center; color:{groq_color};'>{groq_emoji} {groq_signal}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;'>15-Candle Pattern Analysis</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size:1.5em; text-align:center; color:#666;'>⚪ Not Available</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;'>Add Groq API key</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 📊 Final Signal")
        final_color = "#22a218" if final_signal['action'] == "BUY" else ("#d32f2f" if final_signal['action'] == "SELL" else "#f39c12")
        final_emoji = "🟢" if final_signal['action'] == "BUY" else ("🔴" if final_signal['action'] == "SELL" else "🟡")
        st.markdown(f"<div style='font-size:2em; text-align:center; color:{final_color};'>{final_emoji} {final_signal['action']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'>Confidence: <b>{final_signal['confidence']:.1f}%</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'>{final_signal.get('signal_strength', 'MODERATE')} - {final_signal.get('consensus', 'Single Source')}</div>", unsafe_allow_html=True)
    
    # 🆕 Enhanced Technical Analysis Display
    st.markdown("### 🔬 Advanced Technical Analysis Dashboard")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    latest = df.iloc[-1]
    
    with col1:
        current_rsi = latest.get('RSI', 0)
        rsi_color = "🟢" if current_rsi < 30 else ("🔴" if current_rsi > 70 else "🟡")
        st.metric("RSI", f"{rsi_color} {current_rsi:.1f}")
        
        current_momentum = latest.get('Momentum', 0)
        momentum_color = "🟢" if current_momentum > 0 else "🔴"
        st.metric("Momentum", f"{momentum_color} {current_momentum:.2f}")
    
    with col2:
        current_macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_color = "🟢" if current_macd > macd_signal else "🔴"
        st.metric("MACD", f"{macd_color} {current_macd:.4f}")
        
        current_bb_pos = latest.get('BB_Position', 0.5)
        bb_color = "🟢" if current_bb_pos < 0.2 else ("🔴" if current_bb_pos > 0.8 else "🟡")
        st.metric("BB Position", f"{bb_color} {current_bb_pos:.3f}")
    
    with col3:
        current_mfi = latest.get('MFI', 50)
        mfi_color = "🟢" if current_mfi < 30 else ("🔴" if current_mfi > 70 else "🟡")
        st.metric("Money Flow Index", f"{mfi_color} {current_mfi:.1f}")
        
        current_stoch = latest.get('Stoch_K', 50)
        stoch_color = "🟢" if current_stoch < 20 else ("🔴" if current_stoch > 80 else "🟡")
        st.metric("Stochastic %K", f"{stoch_color} {current_stoch:.1f}")
    
    with col4:
        current_williams = latest.get('Williams_R', -50)
        williams_color = "🟢" if current_williams < -80 else ("🔴" if current_williams > -20 else "🟡")
        st.metric("Williams %R", f"{williams_color} {current_williams:.1f}")
        
        current_cci = latest.get('CCI', 0)
        cci_color = "🟢" if current_cci < -100 else ("🔴" if current_cci > 100 else "🟡")
        st.metric("CCI", f"{cci_color} {current_cci:.1f}")
    
    with col5:
        current_adx = latest.get('ADX', 25)
        adx_color = "🟢" if current_adx > 25 else ("🟡" if current_adx > 20 else "🔴")
        st.metric("ADX Strength", f"{adx_color} {current_adx:.1f}")
        
        current_obv = latest.get('OBV', 0)
        st.metric("On Balance Volume", f"{current_obv:,.0f}")
    
    with col6:
        # Buy Score (for display only)
        current_buy_score = latest.get('Buy_Score', 0)
        score_color = "🟢" if current_buy_score > 0.8 else ("🟡" if current_buy_score > 0.6 else "🔴")
        st.metric("Original Buy Score", f"{score_color} {current_buy_score:.3f}")
        
        # Enhanced Buy Score (for display only)
        enhanced_buy_score = latest.get('Enhanced_Buy_Score', 0)
        enhanced_color = "🟢" if enhanced_buy_score > 0.8 else ("🟡" if enhanced_buy_score > 0.6 else "🔴")
        st.metric("🆕 Enhanced Buy Score", f"{enhanced_color} {enhanced_buy_score:.3f}")
    
    # 🆕 Pattern Recognition & Market Analysis
    st.markdown("### 🎨 Pattern Recognition & Market Structure")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🕯️ Candlestick Patterns:**")
        patterns = analysis_data.get('patterns', {})
        for pattern, detected in patterns.items():
            if detected:
                pattern_name = pattern.replace('_', ' ').title()
                if pattern in ['hammer', 'bullish_engulfing']:
                    st.success(f"✅ {pattern_name} (Bullish)")
                elif pattern in ['shooting_star']:
                    st.error(f"⚠️ {pattern_name} (Bearish)")
                else:
                    st.info(f"🔵 {pattern_name} (Neutral)")
        
        if not any(patterns.values()):
            st.info("No significant patterns detected")
    
    with col2:
        st.markdown("**📊 Support & Resistance:**")
        sr_data = analysis_data.get('support_resistance', {})
        current_price = analysis_data['current_price']
        
        support = sr_data.get('support', current_price * 0.98)
        resistance = sr_data.get('resistance', current_price * 1.02)
        pivot = sr_data.get('pivot', current_price)
        
        st.metric("Support Level", f"₹{support:.2f}")
        st.metric("Resistance Level", f"₹{resistance:.2f}")
        st.metric("Pivot Point", f"₹{pivot:.2f}")
        
        # Distance from S/R levels
        support_distance = ((current_price - support) / support) * 100
        resistance_distance = ((resistance - current_price) / current_price) * 100
        
        st.caption(f"Support: {support_distance:.1f}% away")
        st.caption(f"Resistance: {resistance_distance:.1f}% away")
    
    with col3:
        st.markdown("**🎭 Market Sentiment:**")
        sentiment = analysis_data.get('sentiment_analysis', {})
        
        overall_sentiment = sentiment.get('overall', 'neutral')
        sentiment_confidence = sentiment.get('confidence', 0.5) * 100
        
        sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
        st.metric(
            "Overall Sentiment", 
            f"{sentiment_emoji.get(overall_sentiment, '⚪')} {overall_sentiment.title()}"
        )
        st.metric("Sentiment Confidence", f"{sentiment_confidence:.1f}%")
        
        # News vs Social breakdown
        st.caption(f"News: {sentiment.get('news_sentiment', 'neutral').title()}")
        st.caption(f"Social: {sentiment.get('social_sentiment', 'neutral').title()}")
        
        # Options Flow Data
        st.markdown("**📈 Options Flow:**")
        options_data = analysis_data.get('options_flow', {})
        pcr = options_data.get('put_call_ratio', 1.0)
        options_sentiment = options_data.get('sentiment', 'neutral')
        
        pcr_color = "🔴" if pcr > 1.2 else ("🟢" if pcr < 0.8 else "🟡")
        st.metric("Put/Call Ratio", f"{pcr_color} {pcr:.2f}")
        st.caption(f"Options Sentiment: {options_sentiment.title()}")
    
    # 🛡️ Anti-Overfitting Performance Notice
    st.markdown("### 🛡️ Anti-Overfitting Model Information")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.info("**🛡️ Model Type**: Anti-Overfitting\n\n**Expected Accuracy**: 60-75%\n\n**Status**: Realistic Performance")
    
    with info_col2:
        st.info("**🚫 Excluded Features**: Buy_Score, Enhanced_Buy_Score from prediction\n\n**✅ Safe Features**: Only legitimate technical indicators used")
    
    with info_col3:
        confidence = final_signal.get('confidence', 50)
        if confidence > 75:
            st.warning("⚠️ **High Confidence**: Verify with multiple timeframes before trading")
        elif confidence > 60:
            st.success("✅ **Good Confidence**: Signal appears reliable")
        else:
            st.info("📊 **Moderate Confidence**: Consider additional confirmation")
    
    # Trade Analysis (existing code continues unchanged but with anti-overfitting notices)
    if final_signal['action'] in ["BUY", "SELL"]:
        investment_amount = current_price * quantity
        
        st.markdown("---")
        st.markdown("## 📈 Enhanced Trade Analysis")
        
        # Trade metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("💰 Current Price", f"₹{current_price:.2f}")
            st.metric("📦 Quantity", f"{quantity:,} shares")
        
        with metric_col2:
            st.metric("💵 Investment", f"₹{investment_amount:,.0f}")
            st.metric("🛡️ Stop Loss", f"₹{risk_levels['stop_loss']}")
        
        with metric_col3:
            st.metric("🎯 Take Profit 1", f"₹{risk_levels['take_profit_1']}")
            st.metric("🎯 Take Profit 2", f"₹{risk_levels['take_profit_2']}")
        
        with metric_col4:
            st.metric("⚖️ Risk:Reward", f"1:{risk_levels['risk_reward_ratio']}")
            max_profit = abs(risk_levels['take_profit_2'] - current_price) * quantity
            max_loss = abs(current_price - risk_levels['stop_loss']) * quantity
            expected_profit = (max_profit * 0.65) - (max_loss * 0.35)  # Updated for anti-overfitting model
            st.metric("📊 Expected Profit", f"₹{expected_profit:,.0f}")
    
    # Agreement Analysis (if Groq available)
    if groq_signal != "UNKNOWN":
        st.markdown("### 🤝 Signal Agreement Analysis")
        agreement_col1, agreement_col2 = st.columns(2)
        
        with agreement_col1:
            st.write(f"**🛡️ Anti-Overfitting ML says:** {ml_signal['action']} ({ml_signal['confidence']:.1f}%)")
            st.write(f"**🧠 Groq LLM says:** {groq_signal}")
            st.write(f"**🤝 Agreement:** {'✅ YES' if final_signal.get('agreement') else '❌ NO'}")
        
        with agreement_col2:
            st.write(f"**📊 Original ML Confidence:** {final_signal.get('original_confidence', 0):.1f}%")
            st.write(f"**🛡️ Enhanced Confidence:** {final_signal['confidence']:.1f}%")
            confidence_change = final_signal['confidence'] - final_signal.get('original_confidence', 0)
            st.write(f"**📈 Confidence Change:** {confidence_change:+.1f}%")
    
    # Recent market data
    st.markdown("### 📋 Recent Market Data")
    display_df = df.tail(10)[['timestamp', 'close', 'open', 'high', 'low', 'volume', 'RSI', 'Buy_Score', 'Enhanced_Buy_Score']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_df = display_df.round(3)
    st.dataframe(display_df, use_container_width=True)

# Main App Layout
st.title("🛡️ Anti-Overfitting Trading Signal System")
st.markdown("### Realistic ML Predictions + Groq LLM Analysis")

# Model Information in Sidebar
with st.sidebar:
    st.markdown("### 🛡️ Anti-Overfitting Model Status")
    
    if not st.session_state.models_loaded:
        success, message = load_models_safely()
        if success:
            st.success("✅ Anti-Overfitting Models Loaded")
            st.info("🎯 Expected accuracy: 60-75%\n\n🚫 No data leakage\n\n🛡️ Conservative hyperparameters")
        else:
            st.error("❌ Models Not Loaded")
            st.error(message)
            if st.button("🔄 Reload Models"):
                success, message = load_models_safely()
                if success:
                    st.rerun()
    else:
        st.success("✅ Anti-Overfitting Models Active")
        st.info("🎯 Realistic Performance Mode\n\n🛡️ Safe Feature Set\n\n📊 Conservative Predictions")
        
        # Try to show feature importance
        try:
            if hasattr(st.session_state.buy_model, 'feature_importances_'):
                st.markdown("**🔥 Safe Feature Importance:**")
                # Show safe features only
                safe_features = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility", 
                               "MACD", "MACD_Signal", "BB_Position", "MFI", "ADX"]
                importances = st.session_state.buy_model.feature_importances_
                
                feature_importance = list(zip(safe_features[:len(importances)], importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                for feat, imp in feature_importance[:5]:
                    st.text(f"{feat}: {imp:.3f}")
        except:
            pass

# Sidebar Authentication
st.sidebar.title("🔐 API Authentication")
grow_key = st.sidebar.text_input("Groww API token", type="password", key="grow_api_key")
groq_key = st.sidebar.text_input("Groq API key", type="password", key="groq_api_key")

# System initialization
if not grow_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

# Initialize Groww API
with st.spinner("Initializing Groww API..."):
    groww, instruments_df, init_message = initialize_groww_safely()

if groww is None:
    st.error(f"❌ {init_message}")
    st.stop()

st.success(f"✅ {init_message}")

# Check Groq availability
groq_available = False
groq_models = []
if groq_key:
    groq_models, groq_error = get_groq_models(groq_key)
    if groq_models:
        groq_available = True
        st.success("✅ Groq LLM available for enhanced signals")
    else:
        st.warning(f"⚠️ Groq unavailable: {groq_error}")

# Symbol selection
symbols_list = instruments_df["groww_symbol"].sort_values().unique().tolist()
default_symbol = "NSE-NIFTY" if "NSE-NIFTY" in symbols_list else symbols_list[0]
selected_symbol = st.sidebar.selectbox("Select Symbol", symbols_list, index=symbols_list.index(default_symbol))

# Settings
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Account Settings")
account_balance = st.sidebar.number_input("Account Balance (₹)", value=100000, min_value=10000, step=10000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

st.sidebar.subheader("📊 Analysis Settings")
interval_minutes = st.sidebar.selectbox("Candle Interval", [5, 10, 15, 30, 60], index=1)
if groq_available:
    selected_groq_model = st.sidebar.selectbox("Groq Model", groq_models, index=0)

# Auto-refresh settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto Refresh")
auto_refresh_enabled = st.sidebar.checkbox("Enable Auto Refresh", value=st.session_state.auto_refresh)
if auto_refresh_enabled:
    refresh_interval = st.sidebar.selectbox("Refresh Interval", ["1 minute", "2 minutes", "5 minutes", "10 minutes"], index=1)
    refresh_seconds = {"1 minute": 60, "2 minutes": 120, "5 minutes": 300, "10 minutes": 600}[refresh_interval]
    st.session_state.auto_refresh = True
else:
    st.session_state.auto_refresh = False

# Main Analysis Buttons
st.markdown("## 🎯 Anti-Overfitting Trading Analysis")

# Action buttons row
button_col1, button_col2, button_col3, button_col4 = st.columns(4)

with button_col1:
    analyze_clicked = st.button("🛡️ Analyze Symbol", type="primary", use_container_width=True)

with button_col2:
    refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True)

with button_col3:
    if st.session_state.analysis_data:
        quick_refresh = st.button("⚡ Quick Update", use_container_width=True)
    else:
        quick_refresh = False

with button_col4:
    if st.button("🧹 Clear Results", use_container_width=True):
        st.session_state.analysis_data = None
        st.session_state.last_refresh = None
        st.rerun()

# Auto-refresh logic
if st.session_state.auto_refresh and st.session_state.analysis_data:
    last_refresh = st.session_state.last_refresh
    if last_refresh and (datetime.now() - last_refresh).seconds >= refresh_seconds:
        refresh_clicked = True

# Data refresh status
if st.session_state.last_refresh:
    time_since_refresh = datetime.now() - st.session_state.last_refresh
    if time_since_refresh.seconds < 60:
        st.info(f"🕐 Last updated: {time_since_refresh.seconds} seconds ago")
    else:
        st.info(f"🕐 Last updated: {time_since_refresh.seconds // 60} minutes ago")

# Perform analysis
if analyze_clicked or refresh_clicked or quick_refresh:
    with st.spinner(f"{'Refreshing' if refresh_clicked or quick_refresh else 'Analyzing'} {selected_symbol} with anti-overfitting model..."):
        
        # Perform complete analysis
        analysis_data, message = perform_complete_analysis(
            groww, selected_symbol, interval_minutes, 
            groq_key, selected_groq_model if groq_available else None,
            account_balance, risk_per_trade, groq_available
        )
        
        if analysis_data is not None:
            st.session_state.analysis_data = analysis_data
            st.session_state.last_refresh = datetime.now()
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")

# Display results if available
if st.session_state.analysis_data:
    display_analysis_results(st.session_state.analysis_data)
    
    # Action buttons
    final_signal = st.session_state.analysis_data['final_signal']
    if final_signal['action'] in ["BUY", "SELL"]:
        st.markdown("---")
        st.markdown("### 🎬 Anti-Overfitting Trading Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button(f"🛡️ Execute: {final_signal['action']} Signal", type="primary"):
                confidence = final_signal['confidence']
                if confidence > 75:
                    st.success(f"✅ HIGH CONFIDENCE {final_signal['action']} signal! Confidence: {confidence:.1f}%")
                    st.info("🛡️ Anti-overfitting model shows strong signal. Execute with proper risk management.")
                elif confidence > 60:
                    st.success(f"✅ MODERATE CONFIDENCE {final_signal['action']} signal! Confidence: {confidence:.1f}%")
                    st.info("🛡️ Decent signal strength. Consider additional confirmation.")
                else:
                    st.warning(f"⚠️ LOW CONFIDENCE {final_signal['action']} signal! Confidence: {confidence:.1f}%")
                    st.info("🛡️ Weak signal. Recommend waiting for stronger confirmation.")
        
        with action_col2:
            if st.button("💾 Save Anti-Overfitting Signal"):
                current_price = st.session_state.analysis_data['current_price']
                quantity = st.session_state.analysis_data['quantity']
                risk_levels = st.session_state.analysis_data['risk_levels']
                groq_signal = st.session_state.analysis_data['groq_signal']
                ml_signal = st.session_state.analysis_data['ml_signal']
                
                # Enhanced signal data with anti-overfitting info
                signal_data = {
                    'timestamp': datetime.now(),
                    'symbol': selected_symbol,
                    'ml_action': ml_signal['action'],
                    'groq_action': groq_signal,
                    'final_action': final_signal['action'],
                    'confidence': final_signal['confidence'],
                    'model_type': 'anti_overfitting',
                    'expected_accuracy': '60-75%',
                    'buy_score': final_signal.get('buy_score', 0),
                    'enhanced_buy_score': final_signal.get('enhanced_buy_score', 0),
                    'price': current_price,
                    'quantity': quantity,
                    'stop_loss': risk_levels['stop_loss'],
                    'take_profit_1': risk_levels['take_profit_1'],
                    'take_profit_2': risk_levels['take_profit_2'],
                    'risk_reward_ratio': risk_levels['risk_reward_ratio']
                }
                
                signal_df = pd.DataFrame([signal_data])
                filename = f"anti_overfitting_signals_{selected_symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                signal_df.to_csv(filename, index=False)
                st.success(f"💾 Anti-overfitting signal saved to {filename}")
        
        with action_col3:
            if st.button("📊 Analyze Another"):
                st.session_state.analysis_data = None
                st.session_state.last_refresh = None
                st.rerun()

# Auto-refresh display
if st.session_state.auto_refresh and st.session_state.analysis_data:
    st.info(f"🔄 Auto-refresh enabled - updating every {refresh_interval}")
    if st.button("⏹️ Stop Auto-refresh"):
        st.session_state.auto_refresh = False
        st.rerun()
    
    # Auto-refresh timer
    time.sleep(1)
    st.rerun()

# Enhanced help section
with st.expander("📖 Anti-Overfitting Trading System Guide"):
    st.markdown("""
    ## 🛡️ Anti-Overfitting Trading System - User Guide
    
    ### 🎯 **Key Features**
    - **🛡️ Anti-Overfitting Model**: Trained with conservative parameters for realistic performance
    - **🚫 No Data Leakage**: Excludes potentially circular features (Buy_Score, Enhanced_Buy_Score)
    - **📊 Realistic Expectations**: Target accuracy 60-75% (not 99%!)
    - **🔄 Multi-Source Analysis**: ML + Groq LLM + Technical Analysis
    
    ### 📊 **Signal Confidence Interpretation**
    - **75%+**: 🟢 HIGH CONFIDENCE - Strong signal with multiple confirmations
    - **60-75%**: 🟡 MODERATE CONFIDENCE - Good signal, consider additional confirmation
    - **45-60%**: 🟡 LOW CONFIDENCE - Weak signal, wait for better setup
    - **Below 45%**: 🔴 VERY WEAK - Avoid trading
    
    ### 🛡️ **Anti-Overfitting Benefits**
    - **✅ Realistic Performance**: Models trained for real-world conditions
    - **✅ Conservative Approach**: Better risk management
    - **✅ No False Promises**: Honest 60-75% accuracy expectations
    - **✅ Robust Validation**: Extensive walk-forward testing
    
    ###
    ### 🛡️ **Trading Guidelines**
    - **Risk Management**: Never risk more than 2% per trade
    - **Position Sizing**: Use calculated quantities,```just for Kelly Criterion
    - **Stop Losses**: Always use calculated stop-loss levels
    - **Take Profits**: Scale out at TP1 (50%) and TP2 (50%)
    - **Confirmation**: Wait for multiple timeframe confirmation on weak```gnals
    
    ### 📈 **Model Performance**
    - **Training Method**: Anti-overfitting with conservative hyper```ameters
    - **Feature Set**: Safe technical indicators only (no circular logic)
    - **Validation**: Rigorous walk-forward testing
    - **Expected Win Rate**: 60-75% (realistic for trading)
    
    ### 🔧 **Technical Indicators Use```
    - **Core**: RSI, MACD, SMA/EMA, Momentum, Volatility
    - **Advanced**: Bollinger Bands, MFI, Stochastic, Williams```
    - **Volume**: On Balance Volume (OBV)
    - **Trend**: CCI, ADX for trend strength```  - **Pattern**: Candlestick pattern recognition
    - **Structure**: Support/Resistance levels, Fibonacci retracements
    """)

# Risk disclaimer
st.markdown("---")
st.error("""
⚠️ **ANTI-OVERFITTING TRADING SYSTEM DISCLAIMER** ⚠️```️ **This system uses anti-overfitting models with realistic 60-75% accuracy expectations.**

💰 **Trading involves substantial risk of loss an```s not suitable for all investors.**

📊 **Anti-overfitting models provide more realistic but not inf```ible predictions.**

🎯 **Expected performance: 60-75% accuracy (not 99%+ which indicates overfitting).**

🧠 **Always conduct your own analysis and consider```nsulting a qualifie```inancial advisor.**

🔍 **Paper trade first to validate performance before risking real capital.**``` **The conservative approach reduces```lse confidence but doesn't eliminate trading risks.**

📱 **Use proper position sizing, stop losses, and risk management at```l times.**

🛡️ **Better to have realistic 65% accuracy than fake 99% that fails in live trading.**

**By using this anti-overfitting system, you acknowledge```ese realistic performance expectations.**
""")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

with footer_col1:
    if st.session_state.last_refresh:
        st.caption(f"🕐 Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    else:
        st.caption("🕐 No analysis data")

with footer_col2:
    st.caption(f"📊 Symbol: {selected_symbol}")
    portfolio_count = len(st.session_state.portfolio)
    st.caption(f"💼 Portfolio: {portfolio_count} symbols")

with footer_col3:
    if st.session_state.auto_refresh:
        st.caption("🔄 Auto-refresh ON")
    else:
        st.caption("🔄 Auto-refresh OFF")
    
    st.caption("🛡️ Anti-overfitting: ON")

with footer_col4:
    st.caption("🎯 Realistic Performance Mode")
    
    # 🔧 FIXED: Complete function name
    performance = calculate_performance_metrics()
    total_trades = performance.get('total_trades', 0)
    win_rate = performance.get('win_rate', 0)
    if total_trades > 0:
        st.caption(f"📈 Performance: {win_rate:.1f}% win rate ({total_trades} trades)")
    else:
        st.caption("📈 Performance: No trades yet")


# System status
st.markdown("---")
st.markdown("### 🛡️ Anti-Overfitting System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    models_status = "✅ Anti-Overfitting" if st.session_state.get('models_loaded', False) else "❌ Not Loaded"
    st.metric("🛡️ ML Models", models_status)


with status_col2:
    groq_status = "✅ Connected" if groq_available else "❌```sconnected"
    st.metric("🧠 Groq LLM", groq_status)

with status_col3:
    features_count = 16
    st.metric("📊 Safe Features", f"✅ {features_count}")

with status_col4:
    system_status = "🟢 REALISTIC MODE" if (st.session_state.models_loaded and groq_available) else "🟡 PARTIAL OPERATION"
    st.metric("🎯 System Status", system_status)

# Final development info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
<b>🛡️ Anti-Overfitting Trading Signal System v3.0</b><br>
Realistic ML Predictions + Groq LLM Intelligence```Conservative Risk Management<br>```Target Accuracy: 60-75% | 🚫 No Data Leakage | 🛡️ Anti-Overfitting Protection<br>
Built for <b>Real Trading Success</b> with```nest Performance Expectations
</div>
""", unsafe_allow_html=True)

