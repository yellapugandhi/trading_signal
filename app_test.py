import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt_module
from zoneinfo import ZoneInfo
import warnings
import traceback
import joblib
import os
import time
import requests
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🎯 Advanced Trading System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================= SESSION STATE INITIALIZATION =======================
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.buy_model = None
    st.session_state.rr_model = None
    st.session_state.last_refresh = None
    st.session_state.analysis_data = None
    st.session_state.auto_refresh = False
    st.session_state.portfolio = {}
    st.session_state.performance_tracker = []
    st.session_state.alert_settings = {}
    st.session_state.sentiment_cache = {}
    st.session_state.pattern_history = []
    st.session_state.support_resistance_levels = {}

# ======================= MODEL LOADING =======================
@st.cache_data(ttl=300)
def load_models_safely():
    try:
        if os.path.exists("models/buy_model_latest.pkl") and os.path.exists("models/rr_model_latest.pkl"):
            buy_model = joblib.load("models/buy_model_latest.pkl")
            rr_model = joblib.load("models/rr_model_latest.pkl")
            st.session_state.buy_model = buy_model
            st.session_state.rr_model = rr_model
            st.session_state.models_loaded = True
            return True, "🛡️ Anti-overfitting models loaded successfully"
        else:
            return False, "Model files not found. Please run anti_overfitting_retraining.py first."
    except Exception as e:
        return False, f"Error loading models: {str(e)}"

# ======================= GROWW API INTEGRATION =======================
def initialize_groww_safely(grow_key):
    """Initialize Groww API safely"""
    try:
        from growwapi import GrowwAPI
        
        if not grow_key:
            return None, None, "Please enter your Groww API token"
        
        groww = GrowwAPI(grow_key)
        
        try:
            instruments_df = pd.read_csv("instruments.csv")
            groww.instruments = instruments_df
            groww._load_instruments = lambda: None
            groww._download_and_load_instruments = lambda: instruments_df
            
            def get_instrument_by_symbol(symbol):
                matching_instruments = instruments_df[instruments_df['groww_symbol'] == symbol]
                if not matching_instruments.empty:
                    return matching_instruments.iloc[0].to_dict()
                else:
                    st.error(f"Symbol {symbol} not found in instruments")
                    return None
            
            groww.get_instrument_by_groww_symbol = get_instrument_by_symbol
            return groww, instruments_df, "✅ Groww API initialized successfully"
            
        except FileNotFoundError:
            return None, None, "instruments.csv file not found"
        except Exception as e:
            return None, None, f"Error loading instruments: {str(e)}"
            
    except ImportError:
        return None, None, "GrowwAPI not installed. Please install: pip install groww-api"
    except Exception as e:
        return None, None, f"Error initializing Groww API: {str(e)}"

# ---------------- Fetch OI Data Function ----------------

# 🔹 Format expiry safely for matching instrument file
def format_expiry(expiry, available_expiries):
    try:
        expiry = expiry.strip().upper()

        for e in available_expiries:
            if expiry in e.upper():
                return e

        if len(expiry) == 5 and expiry[:2].isdigit():
            day = expiry[:2]
            mon = expiry[2:].capitalize()
            year = str(pd.Timestamp.now().year % 100)
            return f"{day}{mon}{year}"

        return pd.to_datetime(expiry).strftime("%d%b%y")
    except Exception as e:
        raise ValueError(f"Invalid expiry format: {expiry}") from e


# 🔹 Retry wrapper for Groww API
def safe_get_quote(groww, symbols, segment, exchange, retries=3, delay=2):
    results = []
    for symbol in symbols:
        if not symbol or not isinstance(symbol, str):
            st.warning(f"Skipping invalid trading symbol: {symbol}")
            continue
        for attempt in range(retries):
            response = groww.get_quote(segment=segment, trading_symbol=symbol, exchange=exchange)
            if response.get("status") == "SUCCESS":
                results.append(response)
                break
            time.sleep(delay)
        else:
            results.append({"status": "FAILURE", "message": f"Max retries exceeded for {symbol}"})
    return results

# 🔹 Fetch OI changes for CE/PE options
def fetch_option_chain_oi(groww, instruments_df, symbol, expiry, strikes, segment="FNO", exchange="NSE"):
    # Filter matching instruments similarly as before to get trading symbols:
    ce_symbols = []
    pe_symbols = []

    for strike in strikes:
        ce_row = instruments_df[
            (instruments_df['underlying_symbol'] == symbol) &
            (instruments_df['expiry_date'] == pd.to_datetime(expiry, format="%d%b%y")) &
            (instruments_df['strike_price'] == strike) &
            (instruments_df['instrument_type'] == "CE") &
            (instruments_df['segment'] == segment) &
            (instruments_df['exchange'] == exchange)
        ]
        pe_row = instruments_df[
            (instruments_df['underlying_symbol'] == symbol) &
            (instruments_df['expiry_date'] == pd.to_datetime(expiry, format="%d%b%y")) &
            (instruments_df['strike_price'] == strike) &
            (instruments_df['instrument_type'] == "PE") &
            (instruments_df['segment'] == segment) &
            (instruments_df['exchange'] == exchange)
        ]

        if not ce_row.empty:
            ce_symbols.append(ce_row.iloc[0]['trading_symbol'])
        if not pe_row.empty:
            pe_symbols.append(pe_row.iloc[0]['trading_symbol'])

    all_symbols = ce_symbols + pe_symbols
    
    # Call safe_get_quote with all symbols
    all_quotes = safe_get_quote(groww, all_symbols, segment, exchange)

    # Map from trading_symbol to quote
    quotes_dict = {q.get("data", {}).get("trading_symbol", ""): q for q in all_quotes if q.get("status") == "SUCCESS"}

    rows = []
    for strike in strikes:
        ce_sym = next((s for s in ce_symbols if str(strike) in s), None)
        pe_sym = next((s for s in pe_symbols if str(strike) in s), None)

        ce_oi = 0
        pe_oi = 0

        if ce_sym and ce_sym in quotes_dict:
            ce_oi = quotes_dict[ce_sym].get("data", {}).get("openInterest", 0)
        if pe_sym and pe_sym in quotes_dict:
            pe_oi = quotes_dict[pe_sym].get("data", {}).get("openInterest", 0)

        rows.append({
            "strike": strike,
            "CE_OI": ce_oi,
            "PE_OI": pe_oi
        })

    return pd.DataFrame(rows)



#######################
def fetch_latest_candle(groww, symbol, interval_minutes=10, max_candles=50):
    """Fetch latest candle data from Groww API"""
    try:
        selected = groww.get_instrument_by_groww_symbol(symbol)
        if not selected:
            return None
        
        ist = ZoneInfo("Asia/Kolkata")
        now = dt_module.datetime.now(ist)
        
        # Handle weekends
        if now.weekday() >= 5:
            days_back = now.weekday() - 4
            now = now - dt_module.timedelta(days=days_back)
        
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
        
        start_time = end_time - dt_module.timedelta(days=days_needed)
        
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
        
        # Add technical indicators
        df = compute_technical_indicators(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching  {str(e)}")
        return None

# ======================= GROQ LLM INTEGRATION =======================
def get_groq_models(groq_key):
    """Return single preferred Groq model"""
    try:
        import groq
        from groq import Groq
        client = Groq(api_key=groq_key)
        return ["llama-3.3-70b-versatile"], None
    except ImportError:
        return [], "Groq Python lib not installed! Install: pip install groq"
    except Exception as e:
        return [], f"Groq API error: {e}"

def call_groq_llm(df, groq_key, model_name, symbol):
    """Enhanced Groq LLM with 15-candle analysis and pattern recognition"""
    try:
        import groq
        from groq import Groq
        
        # Enhanced: Use 15 candles for better accuracy + pattern context
        recent_data = df.tail(15)
        candles_info = []
        
        for _, row in recent_data.iterrows():
            candle_info = f"Time: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}, "
            candle_info += f"Open: {row['open']:.2f}, High: {row['high']:.2f}, "
            candle_info += f"Low: {row['low']:.2f}, Close: {row['close']:.2f}, "
            candle_info += f"Volume: {row['volume']:,.0f}"
            if 'RSI' in row.index:
                candle_info += f", RSI: {row['RSI']:.1f}"
            if 'MACD' in row.index:
                candle_info += f", MACD: {row['MACD']:.2f}"
            candles_info.append(candle_info)
        
        # Enhanced prompt with price action focus
        prompt = f"""You are an expert price action trader analyzing {symbol}. 

Recent 15-candle 
{chr(10).join(candles_info)}

Analyze using price action methodology:
1. Candlestick patterns (hammer, doji, engulfing, pin bars)
2. Market structure (trend, support/resistance breaks)
3. Volume confirmation with price moves
4. Momentum shifts and divergences

Provide detailed analysis covering:
- Dominant pattern identified
- Market structure assessment
- Volume analysis
- Trading bias (BUY/SELL/HOLD)
- Key levels to watch
- Risk factors

End with: SIGNAL: [BUY/SELL/HOLD]"""
        
        groq_client = Groq(api_key=groq_key)
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        analysis_text = response.choices[0].message.content.strip()
        
        # Extract signal
        signal = 'HOLD'
        if 'SIGNAL: BUY' in analysis_text:
            signal = 'BUY'
        elif 'SIGNAL: SELL' in analysis_text:
            signal = 'SELL'
        
        return {
            'signal': signal,
            'analysis': analysis_text,
            'method': '15-Candle Pattern Analysis'
        }
        
    except Exception as e:
        st.warning(f"Groq LLM error: {e}")
        return {'signal': 'UNKNOWN', 'analysis': f'Error: {str(e)}', 'method': 'Error'}


# ======================= 1. PRICE ACTION FEATURES =======================
def detect_hammer(df):
    """Detect hammer candlestick pattern"""
    if len(df) < 2:
        return False
    
    latest = df.iloc[-1]
    body_size = abs(latest['close'] - latest['open'])
    candle_range = latest['high'] - latest['low']
    
    if candle_range == 0:
        return False
    
    upper_shadow = latest['high'] - max(latest['open'], latest['close'])
    lower_shadow = min(latest['open'], latest['close']) - latest['low']
    
    return (lower_shadow >= 2 * body_size and 
            upper_shadow <= 0.1 * candle_range and 
            body_size <= 0.3 * candle_range)

def detect_doji(df):
    """Detect doji candlestick pattern"""
    if len(df) < 1:
        return False
    
    latest = df.iloc[-1]
    body_size = abs(latest['close'] - latest['open'])
    candle_range = latest['high'] - latest['low']
    
    if candle_range == 0:
        return False
    
    return body_size <= (candle_range * 0.1)

def detect_engulfing(df):
    """Detect engulfing patterns"""
    if len(df) < 2:
        return {'bullish_engulfing': False, 'bearish_engulfing': False}
    
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    bullish = (previous['close'] < previous['open'] and
               current['close'] > current['open'] and
               current['open'] <= previous['close'] and
               current['close'] >= previous['open'])
    
    bearish = (previous['close'] > previous['open'] and
               current['close'] < current['open'] and
               current['open'] >= previous['close'] and
               current['close'] <= previous['open'])
    
    return {'bullish_engulfing': bullish, 'bearish_engulfing': bearish}

def detect_pin_bar(df):
    """Detect pin bar patterns"""
    if len(df) < 1:
        return {'bullish_pin': False, 'bearish_pin': False}
    
    latest = df.iloc[-1]
    body_size = abs(latest['close'] - latest['open'])
    candle_range = latest['high'] - latest['low']
    
    if candle_range == 0:
        return {'bullish_pin': False, 'bearish_pin': False}
    
    upper_shadow = latest['high'] - max(latest['open'], latest['close'])
    lower_shadow = min(latest['open'], latest['close']) - latest['low']
    
    bullish_pin = (lower_shadow >= 2 * body_size and lower_shadow >= 0.6 * candle_range)
    bearish_pin = (upper_shadow >= 2 * body_size and upper_shadow >= 0.6 * candle_range)
    
    return {'bullish_pin': bullish_pin, 'bearish_pin': bearish_pin}

def identify_trend(df, period=20):
    """Identify current trend direction"""
    if len(df) < period:
        return 'insufficient_data'
    
    recent_data = df.tail(period)
    sma_short = recent_data['close'].tail(5).mean()
    sma_long = recent_data['close'].head(5).mean()
    price_momentum = df['close'].iloc[-1] - df['close'].iloc[-period]
    
    if sma_short > sma_long and price_momentum > 0:
        return 'uptrend'
    elif sma_short < sma_long and price_momentum < 0:
        return 'downtrend'
    else:
        return 'sideways'

def analyze_price_action(df):
    """Complete price action analysis"""
    try:
        patterns = {
            'hammer': detect_hammer(df),
            'doji': detect_doji(df),
            'engulfing': detect_engulfing(df),
            'pin_bar': detect_pin_bar(df)
        }
        
        trend = identify_trend(df)
        signal = generate_pa_signal(patterns, trend, df)
        
        return {
            'patterns': patterns,
            'trend': trend,
            'signal': signal['action'],
            'confidence': signal['confidence'],
            'reasoning': signal['reasoning']
        }
        
    except Exception as e:
        return {
            'patterns': {},
            'trend': 'unknown',
            'signal': 'HOLD',
            'confidence': 0,
            'reasoning': f'Error in analysis: {str(e)}'
        }

def generate_pa_signal(patterns, trend, df):
    """Generate price action signal"""
    score = 0
    reasoning = []
    
    # Pattern scoring
    if patterns.get('hammer', False):
        score += 2
        reasoning.append("Bullish hammer pattern")
    
    if patterns.get('engulfing', {}).get('bullish_engulfing', False):
        score += 3
        reasoning.append("Bullish engulfing pattern")
    
    if patterns.get('engulfing', {}).get('bearish_engulfing', False):
        score -= 3
        reasoning.append("Bearish engulfing pattern")
    
    if patterns.get('pin_bar', {}).get('bullish_pin', False):
        score += 2
        reasoning.append("Bullish pin bar")
    
    if patterns.get('pin_bar', {}).get('bearish_pin', False):
        score -= 2
        reasoning.append("Bearish pin bar")
    
    if patterns.get('doji', False):
        reasoning.append("Doji indecision")
    
    # Trend confirmation
    if trend == 'uptrend':
        score += 1
        reasoning.append("Uptrend confirmation")
    elif trend == 'downtrend':
        score -= 1
        reasoning.append("Downtrend confirmation")
    
    # Generate signal
    if score >= 3:
        action = 'BUY'
        confidence = min(score * 15, 85)
    elif score <= -3:
        action = 'SELL'
        confidence = min(abs(score) * 15, 85)
    else:
        action = 'HOLD'
        confidence = 30
    
    return {
        'action': action,
        'confidence': confidence,
        'reasoning': '; '.join(reasoning) if reasoning else 'No clear patterns'
    }

# ======================= TECHNICAL INDICATORS =======================
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
    """Compute On Balance Volume (OBV)"""
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
    """Compute Average Directional Index"""
    try:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr = true_range.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['ADX'] = dx.rolling(period).mean()
        return df
    except:
        df['ADX'] = 25
        return df

def find_support_resistance_levels(df, window=20):
    """Advanced Support & Resistance Detection"""
    try:
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        resistance_points = df[df['high'] == highs]['high'].dropna()
        support_points = df[df['low'] == lows]['low'].dropna()
        
        current_resistance = resistance_points.tail(5).mean() if len(resistance_points) > 0 else df['high'].max()
        current_support = support_points.tail(5).mean() if len(support_points) > 0 else df['low'].min()
        
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        pivot = (high + low + close) / 3
        
        return {
            'support': current_support,
            'resistance': current_resistance,
            'pivot': pivot,
            'strength': len(support_points) + len(resistance_points)
        }
    except:
        return {
            'support': df['low'].min(),
            'resistance': df['high'].max(),
            'pivot': df['close'].iloc[-1],
            'strength': 1
        }

def compute_technical_indicators(df):
    """MASSIVELY ENHANCED: All technical indicators"""
    try:
        df = df.copy()
        
        # Basic indicators
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
        
        # Market regime analysis
        df['Trend_Strength'] = df['SMA_20'].pct_change(5).abs()
        avg_volatility = df['Volatility'].rolling(30).mean()
        recent_volatility = df['Volatility'].rolling(10).mean()
        df['Volatility_Ratio'] = recent_volatility / (avg_volatility + 1e-8)
        
        high_vol_threshold = 1.3
        trend_threshold = 0.02
        
        high_vol_regime = df['Volatility_Ratio'] > high_vol_threshold
        trending_market = df['Trend_Strength'] > trend_threshold
        sideways_market = df['Trend_Strength'] <= 0.01
        
        # Buy Score calculation (your original logic)
        base_weights = {'cond1': 0.40, 'cond2': 0.20, 'cond3': 0.25, 'cond4': 0.15}
        
        w1 = np.full(len(df), base_weights['cond1'])
        w2 = np.full(len(df), base_weights['cond2'])
        w3 = np.full(len(df), base_weights['cond3'])
        w4 = np.full(len(df), base_weights['cond4'])
        
        # Adaptive weights
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
        
        # Original conditions
        cond1 = (df["RSI"] < 45) & (df["Momentum"] > 0)
        cond2 = (df["close"] > df["SMA_10"])
        cond3 = (df["MACD"] > df["MACD_Signal"]) & ((df["MACD"] - df["MACD_Signal"]) > 0)
        cond4 = (df["Momentum"] > df["Momentum"].quantile(0.6)) & (df["RSI"] < 70)
        
        # Buy score calculation
        raw_buy_score = (
            cond1.astype(int) * w1 + 
            cond2.astype(int) * w2 + 
            cond3.astype(int) * w3 + 
            cond4.astype(int) * w4
        )
        
        df['Buy_Score'] = np.where(trending_market.fillna(False), raw_buy_score, raw_buy_score * 0.3)
        
        df['Market_Regime'] = 'Normal'
        df.loc[high_vol_regime.fillna(False), 'Market_Regime'] = 'High Volatility'
        df.loc[sideways_market.fillna(False), 'Market_Regime'] = 'Sideways'
        df.loc[trending_market.fillna(False) & ~high_vol_regime.fillna(False), 'Market_Regime'] = 'Trending'
        
        regime_mapping = {'Normal': 0, 'High Volatility': 1, 'Sideways': 2, 'Trending': 3}
        df['Market_Regime_Num'] = df['Market_Regime'].map(regime_mapping)
        
        # Enhanced indicators
        df = compute_bollinger_bands(df)
        df = compute_money_flow_index(df)  
        df = compute_stochastic_oscillator(df)
        df = compute_williams_r(df)
        df = compute_on_balance_volume(df)
        df = compute_cci(df)
        df = compute_adx(df)
        
        # Enhanced Buy Score
        enhanced_cond5 = (df["MFI"] < 30) & (df["OBV"] > df["OBV"].shift(5))
        enhanced_cond6 = (df["BB_Position"] < 0.2) & (df["close"] > df["close"].shift(1))
        enhanced_cond7 = (df["Stoch_K"] < 20) & (df["Stoch_K"] > df["Stoch_D"])
        enhanced_cond8 = (df["Williams_R"] < -80) & (df["Williams_R"] > df["Williams_R"].shift(1))
        enhanced_cond9 = (df["CCI"] < -100) & (df["CCI"] > df["CCI"].shift(1))
        enhanced_cond10 = (df["ADX"] > 25) & (df["close"] > df["SMA_10"])
        
        df['Enhanced_Buy_Score'] = (
            raw_buy_score * 0.60 + 
            (enhanced_cond5.astype(int) * 0.08 +
             enhanced_cond6.astype(int) * 0.08 +
             enhanced_cond7.astype(int) * 0.08 +
             enhanced_cond8.astype(int) * 0.06 +
             enhanced_cond9.astype(int) * 0.05 +
             enhanced_cond10.astype(int) * 0.05)
        )
        
        # Support/resistance levels
        sr_levels = find_support_resistance_levels(df)
        df['Support_Level'] = sr_levels['support']
        df['Resistance_Level'] = sr_levels['resistance']
        df['Pivot_Level'] = sr_levels['pivot']
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating enhanced technical indicators: {e}")
        return df

# ======================= 2. ML MODEL STRATEGY =======================
def generate_ml_signal(df):
    """ML signal generation compatible with anti-overfitting model"""
    try:
        if not st.session_state.models_loaded:
            return simple_signal_generation(df)
        
        # Ensure Market_Regime_Num exists
        if 'Market_Regime_Num' not in df.columns:
            if 'Market_Regime' in df.columns:
                regime_mapping = {'Normal': 0, 'High_Volatility': 1, 'Sideways': 2, 'Trending': 3}
                df['Market_Regime_Num'] = df['Market_Regime'].map(regime_mapping).fillna(0)
            else:
                df['Market_Regime_Num'] = 0
        
        # Get expected features
        expected_features = st.session_state.buy_model.feature_names_in_
        available_features = [f for f in expected_features if f in df.columns]
        
        if len(available_features) < len(expected_features):
            missing_features = set(expected_features) - set(available_features)
            st.warning(f"⚠️ Missing features: {missing_features}. Using fallback.")
            return simple_signal_generation(df)
        
        # Create input DataFrame
        latest_features = df[expected_features].iloc[-1:].fillna(0)
        
        # ML predictions
        buy_proba = st.session_state.buy_model.predict_proba(latest_features)[0]
        rr_prediction = st.session_state.rr_model.predict(latest_features)[0]
        
        # Calculate confidence
        if len(buy_proba) > 1:
            confidence = buy_proba[1] * 100
        else:
            confidence = buy_proba[0] * 100 if buy_proba[0] > 0.5 else (1 - buy_proba[0]) * 100
        
        confidence = max(min(confidence, 85), 15)
        
        # Determine action
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
            'enhanced_buy_score': df.get('Enhanced_Buy_Score', pd.Series([0.5])).iloc[-1] if 'Enhanced_Buy_Score' in df.columns else 0.5
        }
        
    except Exception as e:
        st.warning(f"🛡️ ML model error: {e}. Using simple signals.")
        return simple_signal_generation(df)

def simple_signal_generation(df):
    """Enhanced fallback signal generation"""
    try:
        latest = df.iloc[-1]
        
        rsi = latest.get('RSI', 50)
        momentum = latest.get('Momentum', 0)
        price = latest.get('close', 0)
        sma_10 = latest.get('SMA_10', price)
        buy_score = latest.get('Buy_Score', 0.5)
        
        # Enhanced indicators
        enhanced_buy_score = latest.get('Enhanced_Buy_Score', buy_score)
        mfi = latest.get('MFI', 50)
        bb_position = latest.get('BB_Position', 0.5)
        stoch_k = latest.get('Stoch_K', 50)
        williams_r = latest.get('Williams_R', -50)
        
        # Scoring system
        base_score = 0
        sell_score = 0
        
        # Original conditions
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
        
        # Advanced indicator scoring
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
        
        # Integrate Enhanced Buy Score
        score_boost = enhanced_buy_score * 3
        final_buy_score = base_score + score_boost
        
        if final_buy_score >= 5:
            action = "BUY"
            confidence = min(final_buy_score * 8, 75)
        elif sell_score >= 4:
            action = "SELL"
            confidence = min(sell_score * 12, 75)
        else:
            action = "HOLD"
            confidence = 45
        
        return {
            'action': action,
            'confidence': confidence,
            'buy_probability': confidence / 100,
            'predicted_rr': 1.5,
            'method': '🛡️ Enhanced Simple Rules',
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

# ======================= 4. INDIVIDUAL INDICATORS ANALYSIS =======================
def analyze_individual_indicators(df):
    """
    Returns:
      - signals_table: List[dict] for DataFrame/table
      - summary: dict with total_score, max_score, bias, action
    """
    latest = df.iloc[-1]
    signals_table = []

    FACTORS = [
        {
            "label": "MACD",
            "weight": "Medium",
            "func": lambda: (
                "BUY" if latest["MACD"] > latest["MACD_Signal"] else "SELL",
                4 if latest["MACD"] > latest["MACD_Signal"] else 2,
                "Medium",
                "Momentum building" if latest["MACD"] > latest["MACD_Signal"] else "Momentum fading"
            )
        },
        {
            "label": "SMA",
            "weight": "Medium",
            "func": lambda: (
                "BUY" if latest["close"] > latest["SMA_10"] else "SELL",
                4 if latest["close"] > latest["SMA_10"] else 2,
                "Medium",
                "Price above average" if latest["close"] > latest["SMA_10"] else "Price below SMA10"
            )
        },
        {
            "label": "Stochastic",
            "weight": "High",
            "func": lambda: (
                "SELL" if latest["Stoch_K"] > 80 else "BUY" if latest["Stoch_K"] < 20 else "HOLD",
                2,
                "High",
                "Overbought zone" if latest["Stoch_K"] > 80 else "Oversold" if latest["Stoch_K"] < 20 else "Neutral"
            )
        },
        {
            "label": "Williams %R",
            "weight": "High",
            "func": lambda: (
                "SELL" if latest["Williams_R"] > -20 else "BUY" if latest["Williams_R"] < -80 else "HOLD",
                2,
                "High",
                "Overbought" if latest["Williams_R"] > -20 else "Oversold" if latest["Williams_R"] < -80 else "Neutral"
            )
        },
        {
            "label": "CCI",
            "weight": "Medium",
            "func": lambda: (
                "SELL" if latest["CCI"] > 100 else "BUY" if latest["CCI"] < -100 else "HOLD",
                2,
                "Medium",
                "Possible reversal" if abs(latest["CCI"]) > 100 else "Neutral"
            )
        },
        {
            "label": "ADX",
            "weight": "High",
            "func": lambda: (
                "Strong Trend" if latest["ADX"] > 25 else "Weak Trend",
                2 if latest["ADX"] > 25 else 1,
                "High",
                "ADX>25 (trend building)" if latest["ADX"] > 25 else "No strong bias"
            )
        },
        {
            "label": "Bollinger",
            "weight": "Medium",
            "func": lambda: (
                "SELL" if latest["BB_Position"] > 0.8 else "BUY" if latest["BB_Position"] < 0.2 else "HOLD",
                2,
                "Medium",
                "Overextended" if latest["BB_Position"] > 0.8 else "Oversold" if latest["BB_Position"] < 0.2 else "Neutral"
            )
        },
        {
            "label": "MFI",
            "weight": "Medium",
            "func": lambda: (
                "SELL" if latest["MFI"] > 70 else "BUY" if latest["MFI"] < 30 else "HOLD",
                2,
                "Medium",
                "Overbought" if latest["MFI"] > 70 else "Oversold" if latest["MFI"] < 30 else "Neutral"
            )
        }
    ]

    total_score = 0
    max_score = 0

    for f in FACTORS:
        try:
            sig, score, _, notes = f["func"]()
            weight_num = 2 if f["weight"] == "High" else 1
            weighted_score = score * weight_num
            signals_table.append({
                "Factor": f["label"],
                "Signal": sig,
                "Weight": f["weight"],
                "Score": score,
                "WeightedScore": weighted_score,
                "Notes": notes
            })
            total_score += weighted_score
            max_score += 5 * weight_num
        except Exception as e:
            signals_table.append({
                "Factor": f["label"], "Signal": "?", "Weight": f["weight"], "Score": "?", "WeightedScore": 0, "Notes": str(e)
            })
            max_score += 5 * (2 if f["weight"] == "High" else 1)

    # Bias/summary logic
    if total_score >= max_score * 0.8:
        bias = "Bullish (strong conviction)"
        action = "Aggressive: Enter or scale in quickly"
    elif total_score >= max_score * 0.65:
        bias = "Cautious Bullish"
        action = "Enter with risk control / partial size"
    elif total_score >= max_score * 0.5:
        bias = "Neutral"
        action = "Wait for confirmation or stand aside"
    else:
        bias = "No action Zone"
        action = "NO action"

    summary = {
        "total_score": total_score,
        "max_score": max_score,
        "bias": bias,
        "action": action
    }
    return signals_table, summary


# ======================= 5. SUPPORT & RESISTANCE MULTI-TIMEFRAME =======================
def calculate_support_resistance_multiTF(df):
    """Calculate support and resistance for multiple timeframes"""
    try:
        # Current timeframe S&R
        current_sr = find_support_resistance_levels(df)
        
        # Simulate 4H and 1D levels
        recent_30 = df.tail(30)  # ~4H equivalent
        recent_144 = df.tail(144)  # ~1D equivalent
        
        sr_4h = find_support_resistance_levels(recent_30)
        sr_1d = find_support_resistance_levels(recent_144)
        
        return {
            'current_tf': current_sr,
            '4H': sr_4h,
            '1D': sr_1d,
            'confluence_levels': find_confluence_levels([current_sr, sr_4h, sr_1d])
        }
        
    except Exception as e:
        return {
            'current_tf': {'support': 0, 'resistance': 0},
            '4H': {'support': 0, 'resistance': 0},
            '1D': {'support': 0, 'resistance': 0},
            'confluence_levels': [],
            'error': str(e)
        }

def find_confluence_levels(sr_list):
    """Find confluence between different timeframe levels"""
    try:
        all_levels = []
        
        for sr in sr_list:
            if 'support' in sr and sr['support'] > 0:
                all_levels.append(sr['support'])
            if 'resistance' in sr and sr['resistance'] > 0:
                all_levels.append(sr['resistance'])
        
        if not all_levels:
            return []
        
        confluence = []
        tolerance = np.std(all_levels) * 0.5
        
        for level in set(all_levels):
            close_levels = [l for l in all_levels if abs(l - level) <= tolerance]
            if len(close_levels) >= 2:
                confluence.append({
                    'level': round(np.mean(close_levels), 2),
                    'strength': len(close_levels)
                })
        
        return sorted(confluence, key=lambda x: x['strength'], reverse=True)[:3]
        
    except Exception as e:
        return []

# ======================= 6. ENTRY & RISK MANAGEMENT =======================
def calculate_risk_levels(df, action, current_price, target_rr=1.5):
    """Enhanced risk level calculation"""
    latest = df.iloc[-1]
    atr = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 1.0
    atr = max(atr, 1.0)
    buffer = atr * 1.2

    support_level = latest.get('Support_Level', current_price - buffer)
    resistance_level = latest.get('Resistance_Level', current_price + buffer)

    if action == "BUY":
        stop_loss = max(current_price - buffer, support_level * 0.98)
        price_risk = abs(current_price - stop_loss)
        max_profit = price_risk * target_rr
        
        take_profit_1 = current_price + max_profit * 0.6
        take_profit_2 = current_price + max_profit
        
        take_profit_1 = min(take_profit_1, resistance_level * 0.98)
        take_profit_2 = min(take_profit_2, resistance_level)
        
    elif action == "SELL":
        stop_loss = min(current_price + buffer, resistance_level * 1.02)
        price_risk = abs(stop_loss - current_price)
        max_profit = price_risk * target_rr
        
        take_profit_1 = current_price - max_profit * 0.6
        take_profit_2 = current_price - max_profit
        
        take_profit_1 = max(take_profit_1, support_level) if take_profit_1 > support_level else take_profit_1
        take_profit_2 = max(take_profit_2, support_level) if take_profit_2 > support_level else take_profit_2
        
    else:
        stop_loss = current_price - buffer
        take_profit_1 = current_price + buffer * 0.5
        take_profit_2 = current_price + buffer

    # Ensure correct TP ordering
    if action == "SELL":
        if take_profit_1 < take_profit_2:
            take_profit_1, take_profit_2 = take_profit_2, take_profit_1
    elif action == "BUY":  
        if take_profit_1 > take_profit_2:
            take_profit_1, take_profit_2 = take_profit_2, take_profit_1

    # Round values
    stop_loss = round(stop_loss, 2)
    take_profit_1 = round(take_profit_1, 2) 
    take_profit_2 = round(take_profit_2, 2)
    
    # Calculate risk:reward
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

def calculate_position_size(current_price, stop_loss, account_balance, risk_per_trade):
    """Enhanced position sizing"""
    try:
        if stop_loss == 0 or current_price == stop_loss:
            return 0
        
        risk_amount = account_balance * (risk_per_trade / 100)
        price_risk = abs(current_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = int(risk_amount / price_risk)
        max_shares_by_value = int((account_balance * 0.10) / current_price)
        
        return min(position_size, max_shares_by_value, 1000)
        
    except:
        return 0

# ======================= DISPLAY FUNCTIONS =======================
def display_price_action_section(pa_data):
    """Display 1. Price Action Strategy"""
    st.header("1. 🕯️ Price Action Strategy")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_color = "#22a218" if pa_data['signal'] == "BUY" else ("#d32f2f" if pa_data['signal'] == "SELL" else "#f39c12")
        signal_emoji = "🟢" if pa_data['signal'] == "BUY" else ("🔴" if pa_data['signal'] == "SELL" else "🟡")
        st.markdown(f"<div style='text-align:center; font-size:2em; color:{signal_color};'>{signal_emoji} <strong>{pa_data['signal']}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-weight:bold;'>Confidence: {pa_data['confidence']:.0f}%</div>", unsafe_allow_html=True)
    
    with col2:
        trend_color = "#22a218" if pa_data['trend'] == "uptrend" else ("#d32f2f" if pa_data['trend'] == "downtrend" else "#f39c12")
        st.markdown("**Market Trend**")
        st.markdown(f"<div style='color:{trend_color}; font-size:1.2em;'>📈 {pa_data['trend'].upper()}</div>", unsafe_allow_html=True)
    
    with col3:
        patterns = pa_data['patterns']
        pattern_count = sum([1 for k, v in patterns.items() if v is True or (isinstance(v, dict) and any(v.values()))])
        st.markdown("**Patterns Found**")
        st.markdown(f"<div style='font-size:1.5em; text-align:center;'>{pattern_count}</div>", unsafe_allow_html=True)
    
    # Price Action Features
    with st.expander("📊 Price Action Features Detail"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Candlestick Patterns:**")
            for pattern, detected in patterns.items():
                if isinstance(detected, dict):
                    for sub_pattern, sub_detected in detected.items():
                        icon = "✅" if sub_detected else "❌"
                        st.write(f"{icon} {sub_pattern.replace('_', ' ').title()}")
                else:
                    icon = "✅" if detected else "❌"
                    st.write(f"{icon} {pattern.replace('_', ' ').title()}")
        
        with col2:
            st.markdown("**Analysis Summary:**")
            st.write(f"**Reasoning:** {pa_data['reasoning']}")
            st.write(f"**Trend Direction:** {pa_data['trend']}")
            st.write(f"**Signal Strength:** {pa_data['confidence']:.0f}%")

def display_ml_section(ml_data):
    """Display 2. ML Model Strategy"""
    st.header("2. 🤖 ML Model Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        signal_color = "#22a218" if ml_data['action'] == "BUY" else ("#d32f2f" if ml_data['action'] == "SELL" else "#f39c12")
        signal_emoji = "🟢" if ml_data['action'] == "BUY" else ("🔴" if ml_data['action'] == "SELL" else "🟡")
        st.markdown(f"<div style='text-align:center; font-size:2em; color:{signal_color};'>{signal_emoji} <strong>{ml_data['action']}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-weight:bold;'>Confidence: {ml_data['confidence']:.0f}%</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Model Details:**")
        st.write(f"**Method:** {ml_data['method']}")
        st.write(f"**Buy Score:** {ml_data.get('buy_score', 0):.3f}")
        st.write(f"**Enhanced Score:** {ml_data.get('enhanced_buy_score', 0):.3f}")

def display_groq_section(groq_data):
    """Display 3. Groq Model Analysis"""
    st.header("3. 🧠 Groq Model")
    
    if groq_data['signal'] != 'UNKNOWN':
        col1, col2 = st.columns([1, 2])
        
        with col1:
            signal_color = "#22a218" if groq_data['signal'] == "BUY" else ("#d32f2f" if groq_data['signal'] == "SELL" else "#f39c12")
            signal_emoji = "🟢" if groq_data['signal'] == "BUY" else ("🔴" if groq_data['signal'] == "SELL" else "🟡")
            st.markdown(f"<div style='text-align:center; font-size:2em; color:{signal_color};'>{signal_emoji} <strong>{groq_data['signal']}</strong></div>", unsafe_allow_html=True)
            st.write(f"**Method:** {groq_data['method']}")
        
        with col2:
            st.markdown("**AI Analysis:**")
            st.text_area("", groq_data['analysis'], height=150, disabled=True)
    else:
        st.warning("⚠️ Groq analysis not available")
        st.write(groq_data['analysis'])

def display_indicators_section(signals, summary):
    st.header("4. All Indicators: Multi-factor Table & Score")
    st.markdown(f"**🧠 Score (out of {summary['max_score']}):** {summary['total_score']:.0f}")
    st.markdown(f"**Bias:** {summary['bias']}")
    st.markdown(f"**Action:** {summary['action']}")
    st.dataframe(pd.DataFrame(signals))


def display_support_resistance_section(sr_data):
    """Display 5. Support and Resistance"""
    st.header("5. 🏗️ Support and Resistance - 1 Day, 4hrs Value")
    
    if 'error' in sr_data:
        st.error(f"S&R Error: {sr_data['error']}")
    
    # Multi-timeframe levels display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1D Timeframe:**")
        if '1D' in sr_data:
            support_1d = sr_data['1D'].get('support', 0)
            resistance_1d = sr_data['1D'].get('resistance', 0)
            st.metric("Support", f"₹{support_1d:.2f}")
            st.metric("Resistance", f"₹{resistance_1d:.2f}")
    
    with col2:
        st.markdown("**4H Timeframe:**")
        if '4H' in sr_data:
            support_4h = sr_data['4H'].get('support', 0)
            resistance_4h = sr_data['4H'].get('resistance', 0)
            st.metric("Support", f"₹{support_4h:.2f}")
            st.metric("Resistance", f"₹{resistance_4h:.2f}")
    
    with col3:
        st.markdown("**Current TF:**")
        if 'current_tf' in sr_data:
            support_curr = sr_data['current_tf'].get('support', 0)
            resistance_curr = sr_data['current_tf'].get('resistance', 0)
            st.metric("Support", f"₹{support_curr:.2f}")
            st.metric("Resistance", f"₹{resistance_curr:.2f}")
    
    # Confluence levels
    if 'confluence_levels' in sr_data and sr_data['confluence_levels']:
        st.markdown("**📍 Key Confluence Levels:**")
        for i, level in enumerate(sr_data['confluence_levels'][:3], 1):
            st.write(f"{i}. ₹{level['level']:.2f} (Strength: {level['strength']})")

def display_entry_stop_loss_section(analysis_data):
    """Display 6. Entry Level and Stop Loss"""
    st.header("6. 🎯 Entry Level and Stop Loss")
    
    current_price = analysis_data['current_price']
    risk_levels = analysis_data['risk_levels']
    quantity = analysis_data['quantity']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Current Price", f"₹{current_price:.2f}")
        st.metric("📊 Quantity", f"{quantity} shares")
    
    with col2:
        st.metric("🛑 Stop Loss", f"₹{risk_levels['stop_loss']:.2f}")
        st.metric("🎯 Take Profit 1", f"₹{risk_levels['take_profit_1']:.2f}")
    
    with col3:
        st.metric("🎯 Take Profit 2", f"₹{risk_levels['take_profit_2']:.2f}")
        st.metric("⚖️ Risk:Reward", f"1:{risk_levels['risk_reward_ratio']:.1f}")
    
    # Additional risk info
    with st.expander("🔍 Detailed Risk Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**ATR:** {risk_levels['atr']:.2f}")
            st.write(f"**Support Level:** ₹{risk_levels['support_level']:.2f}")
        
        with col2:
            st.write(f"**Resistance Level:** ₹{risk_levels['resistance_level']:.2f}")
            risk_amount = abs(current_price - risk_levels['stop_loss']) * quantity
            reward_amount = abs(risk_levels['take_profit_2'] - current_price) * quantity
            st.write(f"**Risk Amount:** ₹{risk_amount:.2f}")
            st.write(f"**Reward Amount:** ₹{reward_amount:.2f}")

def strong_price_structure_override(df):
    """
    Check if the last few candles confirm a strong bullish structure:
    - Higher lows and higher highs over last 4 candles
    - Price closing above recent resistance level
    """
    if len(df) < 10:
        return False
    
    recent = df.tail(6)
    highs = recent['high']
    lows = recent['low']
    closes = recent['close']
    resistance = recent['Resistance_Level'].iloc[-1]
    
    # Check for higher lows and higher highs
    hl = all(lows.iloc[i] > lows.iloc[i-1] for i in range(1, len(lows)))
    hh = all(highs.iloc[i] > highs.iloc[i-1] for i in range(1, len(highs)))
    
    # Check last close above resistance
    close_above_resistance = closes.iloc[-1] > resistance
    
    return hl and hh and close_above_resistance

# ------------------- NEW FEATURES START --------------------

# 1. Real-time Email Alert Demo (Replace with actual credentials and SMTP config)
def send_email_alert(subject, body, recipient_email):
    sender_email = "youremail@example.com"
    sender_password = "yourpassword"  # Use environment vars / secrets in prod

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Using Gmail SMTP as example
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email alert failed: {e}")
        return False

# 2. Simple Backtesting Logic for a strategy (dummy example)
def backtest_strategy(df, entry_col='Buy_Score', threshold=0.7):
    df = df.copy()
    df['Position'] = 0
    df.loc[df[entry_col] > threshold, 'Position'] = 1
    df['Returns'] = df['close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1).fillna(0)
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    return df

# 3. Interactive Candlestick chart with overlays using Plotly
def plot_candlestick_with_indicators(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    )])

    # Add moving averages
    if 'SMA_10' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_10'], mode='lines', name='SMA 10'))
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_20'], mode='lines', name='SMA 20'))

    # Add Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_Upper'], mode='lines', line=dict(color='rgba(255,0,0,0.2)'), showlegend=True, name='BB Upper'))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_Lower'], mode='lines', line=dict(color='rgba(0,255,0,0.2)'), showlegend=True, name='BB Lower'))

    fig.update_layout(title='Candlestick Chart with Indicators', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# 4. Sentiment Analysis placeholder (modify with real API later)
def fetch_sentiment_score(symbol):
    # Dummy stub: returns random sentiment between -1 and 1
    return np.random.uniform(-1, 1)

# 5. Basic Portfolio Tracking in session state
def update_portfolio(position, price, qty):
    portfolio = st.session_state.get('portfolio', {})
    if position == 'buy':
        portfolio['qty'] = portfolio.get('qty', 0) + qty
        portfolio['avg_price'] = ((portfolio.get('avg_price', 0) * portfolio.get('qty', 0)) + price * qty) / (portfolio.get('qty', 0) + qty)
    elif position == 'sell':
        portfolio['qty'] = max(portfolio.get('qty', 0) - qty, 0)
    st.session_state['portfolio'] = portfolio

def display_portfolio():
    portfolio = st.session_state.get('portfolio', {})
    st.write("### Portfolio Summary:")
    st.write(portfolio)

# 6. Model Explainability placeholder (needs actual SHAP or similar)
def show_model_explanation():
    st.write("### Model Feature Importance (Example)")
    st.bar_chart({
        'Feature': ['Momentum', 'RSI', 'MACD', 'Volume', 'Volatility'],
        'Importance': [0.3, 0.25, 0.2, 0.15, 0.1]
    })

# 7. Multi-symbol Watchlist management
def add_symbol_to_watchlist(symbol):
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = []
    if symbol not in st.session_state['watchlist']:
        st.session_state['watchlist'].append(symbol)

def display_watchlist():
    watchlist = st.session_state.get('watchlist', [])
    st.sidebar.write("### Watchlist")
    for sym in watchlist:
        st.sidebar.write(sym)

# 8. Parameter tuning panel by user input
def parameter_tuning():
    st.sidebar.write("### Tuning Parameters")
    period_rsi = st.sidebar.slider("RSI Period", 5, 50, 14)
    period_sma = st.sidebar.slider("SMA Period", 5, 50, 20)
    return period_rsi, period_sma

# 9. Dynamic risk management adjustment
def adjust_risk_based_on_volatility(df, base_risk=0.01):
    vol = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 1.0
    adjusted_risk = base_risk * (vol / df['Volatility'].mean())
    return max(min(adjusted_risk, 0.05), 0.005)  # cap risk between 0.5% to 5%

# 10. Loading spinner during computations
from contextlib import contextmanager
@contextmanager
def spinner(message):
    with st.spinner(message):
        yield

# ------------------- NEW FEATURES END -----------------------

# ======================= MAIN DASHBOARD =======================
# --- START: Beautification and UI Enhancements below ---

# Custom CSS for headers and color
st.markdown("""
<style>
/* Colorful metrics and section headers */
.big-header {
    font-size: 2.7rem !important;
    color: #1C2833 !important;
    font-weight: 800;
    letter-spacing: 0.03em;
    margin-bottom: 6px;
}
.section-title {
    font-size: 1.6rem !important;
    color: #2874A6 !important;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 4px;
}
.metric-value, .stMetric {
    font-weight: bold !important;
    font-size: 2.1rem !important;
}
.st-bw, .stButton>button {
    border-radius: 8px;
    background: linear-gradient(90deg,#E8DAEF, #D4EFDF,#F9E79F);
    color: #222 !important;
    font-size: 1.21rem !important;
}
.st-emotion-cache-0 {  /* Reduce sidebar padding, more content fits */
    padding-top: 1rem !important;
}
.stDataFrame {background: #FDFEFE !important;}
.stExpanderHeader {font-size:1.13rem !important; color:#CB4335;}
/* Tab headers more visible */
.stTabs [data-baseweb="tab"] {
    background-color: #E8F8F5 !important;
    color: #007474 !important;
    font-weight: bold !important;
    border-radius: 12px 12px 0 0;
    font-size: 1.0rem;
    padding: 4px 26px 4px 26px;
}
.stTabs [aria-selected="true"] {
    background: #F9E79F !important;
    color: #B03A2E !important;
}
</style>
""", unsafe_allow_html=True)
# --- END: Beautification and UI Enhancements above ---

# ======================= MAIN DASHBOARD =======================
def main_dashboard():
    st.title("🎯 Trading System")
    st.markdown("created by Gandhi Yellapu")

    with st.sidebar:
        st.header("⚙️ Configuration")
        grow_api_key = st.text_input("Groww API Key", type="password", key="grow_api_key")
        groq_key = st.text_input("Groq API Key", type="password", key="groq_key")
        account_balance = st.number_input("Account Balance (₹)", min_value=10000, value=100000, step=10000)
        risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        symbol = st.selectbox("Select Symbol", ["NSE-NIFTY", "NSE-BANKNIFTY", "NSE-RELIANCE", "NSE-TCS", "NSE-INFY", "NSE-HDFC"], index=0)
        interval_minutes = st.selectbox("Candle Interval", [1,5,10,15,30,60], index=0)
        analyze_btn = st.button("🎯 Complete Analysis", type="primary", use_container_width=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.experimental_rerun()

    # Initialize APIs & models
    if 'groww' not in st.session_state:
        st.session_state.groww = None
    if grow_api_key and (not st.session_state.groww):
        groww, instruments_df, msg = initialize_groww_safely(grow_api_key)
        st.session_state.groww = groww
        if groww:
            st.success(msg)
        else:
            st.warning(msg)

    models_loaded, model_msg = load_models_safely()
    if not models_loaded:
        st.warning(model_msg)
    else:
        st.success(model_msg)

    # Run analysis on button click
    if analyze_btn and st.session_state.groww:
        groww = st.session_state.groww
        try:
            df = fetch_latest_candle(groww, symbol, interval_minutes, 100)
            if df is None or len(df) < 20:
                st.error("Insufficient candle data")
                return
            current_price = df['close'].iloc[-1]

            pa_analysis = analyze_price_action(df)
            ml_analysis = generate_ml_signal(df)
            groq_analysis = None
            if groq_key:
                groq_analysis = call_groq_llm(df, groq_key, "llama-3.3-70b-versatile", symbol)
            else:
                groq_analysis = {"signal":"No Groq Key", "analysis":"", "method":""}
            signals, summary = analyze_individual_indicators(df)
            sr_levels = calculate_support_resistance_multiTF(df)
            risk_levels = calculate_risk_levels(df, ml_analysis['action'], current_price)
            quantity = calculate_position_size(current_price, risk_levels['stop_loss'], account_balance, risk_per_trade)

            st.session_state.analysis_data = {
                'df': df,
                'pa_analysis': pa_analysis,
                'ml_analysis': ml_analysis,
                'groq_analysis': groq_analysis,
                'signals': signals,
                'summary': summary,
                'sr_levels': sr_levels,
                'risk_levels': risk_levels,
                'current_price': current_price,
                'quantity': quantity,
                'timestamp': dt_module.datetime.now(),
                'symbol': symbol
            }

            st.success(f"Analysis completed successfully for {symbol}.")

        except Exception as e:
            st.error(f"Analysis error: {e}")
            traceback.print_exc()

    # Display results & OI table
    if st.session_state.get('analysis_data'):
        data = st.session_state.analysis_data

        st.info(f"Last updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} for {data['symbol']}")

        display_price_action_section(data['pa_analysis']); st.markdown("---")
        display_ml_section(data['ml_analysis']); st.markdown("---")
        display_groq_section(data['groq_analysis']); st.markdown("---")
        display_indicators_section(data['signals'], data['summary']); st.markdown("---")
        display_support_resistance_section(data['sr_levels']); st.markdown("---")
        display_entry_stop_loss_section(data); st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"₹{data['current_price']:.2f}")
        col2.metric("RSI", f"{data['df']['RSI'].iloc[-1]:.1f}")
        col3.metric("MACD", f"{data['df']['MACD'].iloc[-1]:.3f}")
        col4.metric("Volume", f"{data['df']['volume'].iloc[-1]:,}")

if __name__ == "__main__":
    main_dashboard()
