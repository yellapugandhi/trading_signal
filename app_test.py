import sys
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

# 🔧 FIX: Increase recursion limit to prevent Groww API errors
sys.setrecursionlimit(3000)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🛡️ Anti-Overfitting Trading Signal System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 🔧 FIX: Safe session state initialization
def init_session_state():
    """Initialize session state safely to prevent recursion"""
    defaults = {
        'models_loaded': False,
        'buy_model': None,
        'rr_model': None,
        'last_refresh': None,
        'analysis_data': None,
        'auto_refresh': False,
        'portfolio': {},
        'performance_tracker': [],
        'alert_settings': {},
        'sentiment_cache': {},
        'pattern_history': [],
        'support_resistance_levels': {},
        'groww_initialized': False,
        'groww_api': None,
        'instruments_df': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
init_session_state()

@st.cache_data(ttl=300)
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

# 🔧 FIXED: Safe Groww API initialization without recursion
@st.cache_resource
def initialize_groww_safely():
    """FIXED: Initialize Groww API safely without recursion"""
    try:
        from growwapi import GrowwAPI
    except ImportError:
        return None, None, "GrowwAPI not installed. Please install: pip install groww-api"
    
    grow_key = st.session_state.get("grow_api_key", "")
    if not grow_key:
        return None, None, "Please enter your Groww API token"
    
    try:
        # Simple initialization without dangerous overrides
        groww = GrowwAPI(grow_key)
        instruments_df = None
        
        # Try to load instruments file
        try:
            instruments_df = pd.read_csv("instruments.csv")
            
            # Safe instrument lookup function
            def safe_get_instrument_by_symbol(symbol):
                try:
                    matching_instruments = instruments_df[instruments_df['groww_symbol'] == symbol]
                    if not matching_instruments.empty:
                        return matching_instruments.iloc[0].to_dict()
                    else:
                        st.error(f"Symbol {symbol} not found in instruments")
                        return None
                except Exception as e:
                    st.error(f"Error finding symbol {symbol}: {e}")
                    return None
            
            # Assign safe function
            groww.get_instrument_by_groww_symbol = safe_get_instrument_by_symbol
            
        except FileNotFoundError:
            st.warning("instruments.csv file not found. Some features may be limited.")
        
        return groww, instruments_df, "Groww API initialized successfully"
        
    except Exception as e:
        return None, None, f"Error initializing Groww API: {str(e)}"

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
    """Enhanced Groq LLM with 15-candle analysis"""
    try:
        import groq
        from groq import Groq
        
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
        
        prompt = f"""Analyze this {len(candles_info)}-candle data for {symbol}:

{chr(10).join(candles_info)}

Rules: Respond with exactly ONE word: BUY, SELL, or HOLD"""
        
        groq_client = Groq(api_key=groq_key)
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        return result if result in ["BUY", "SELL", "HOLD"] else "UNCLEAR"
            
    except Exception as e:
        st.warning(f"Groq LLM error: {e}")
        return "UNKNOWN"

# Technical Indicators Functions
def compute_bollinger_bands(df, period=20, std_dev=2):
    """Compute Bollinger Bands"""
    df['BB_Middle'] = df['close'].rolling(period).mean()
    bb_std = df['close'].rolling(period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * std_dev)
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df

def compute_money_flow_index(df, period=14):
    """Compute Money Flow Index"""
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
    """Compute Stochastic Oscillator"""
    lowest_low = df['low'].rolling(k_period).min()
    highest_high = df['high'].rolling(k_period).max()
    
    df['Stoch_K'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    df['Stoch_D'] = df['Stoch_K'].rolling(d_period).mean()
    return df

def compute_williams_r(df, period=14):
    """Compute Williams %R"""
    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()
    
    df['Williams_R'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    return df

def compute_technical_indicators(df):
    """Complete technical indicators calculation"""
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
        
        # Trend analysis
        df['Trend_Strength'] = df['SMA_20'].pct_change(5).abs()
        
        avg_volatility = df['Volatility'].rolling(30).mean()
        recent_volatility = df['Volatility'].rolling(10).mean()
        df['Volatility_Ratio'] = recent_volatility / (avg_volatility + 1e-8)
        
        # Market regime
        high_vol_regime = df['Volatility_Ratio'] > 1.3
        trending_market = df['Trend_Strength'] > 0.02
        sideways_market = df['Trend_Strength'] <= 0.01
        
        df['Market_Regime'] = 'Normal'
        df.loc[high_vol_regime.fillna(False), 'Market_Regime'] = 'High_Volatility'
        df.loc[sideways_market.fillna(False), 'Market_Regime'] = 'Sideways'
        df.loc[trending_market.fillna(False) & ~high_vol_regime.fillna(False), 'Market_Regime'] = 'Trending'
        
        # Market regime numerical
        regime_mapping = {'Normal': 0, 'High_Volatility': 1, 'Sideways': 2, 'Trending': 3}
        df['Market_Regime_Num'] = df['Market_Regime'].map(regime_mapping).fillna(0)
        
        # Advanced indicators
        df = compute_bollinger_bands(df)
        df = compute_money_flow_index(df)
        df = compute_stochastic_oscillator(df)
        df = compute_williams_r(df)
        
        # Buy score calculation
        cond1 = (df["RSI"] < 45) & (df["Momentum"] > 0)
        cond2 = (df["close"] > df["SMA_10"])
        cond3 = (df["MACD"] > df["MACD_Signal"])
        cond4 = (df["Momentum"] > df["Momentum"].quantile(0.6)) & (df["RSI"] < 70)
        
        df['Buy_Score'] = (
            cond1.astype(int) * 0.35 + 
            cond2.astype(int) * 0.25 + 
            cond3.astype(int) * 0.25 + 
            cond4.astype(int) * 0.15
        )
        
        # Enhanced buy score
        enhanced_cond5 = (df["MFI"] < 30)
        enhanced_cond6 = (df["BB_Position"] < 0.2) & (df["close"] > df["close"].shift(1))
        enhanced_cond7 = (df["Stoch_K"] < 20) & (df["Stoch_K"] > df["Stoch_D"])
        enhanced_cond8 = (df["Williams_R"] < -80) & (df["Williams_R"] > df["Williams_R"].shift(1))
        
        df['Enhanced_Buy_Score'] = (
            df['Buy_Score'] * 0.60 + 
            (enhanced_cond5.astype(int) * 0.10 +
             enhanced_cond6.astype(int) * 0.10 +
             enhanced_cond7.astype(int) * 0.10 +
             enhanced_cond8.astype(int) * 0.10)
        )
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return df

# 🔧 FIXED: ML Signal Generation with proper error handling
def generate_ml_signal(df):
    """FIXED: ML signal generation compatible with anti-overfitting model"""
    try:
        if not st.session_state.get('models_loaded', False):
            return simple_signal_generation(df)
        
        # Ensure required features exist
        required_features = ['SMA_10', 'EMA_10', 'RSI', 'Momentum', 'Volatility', 
                           'Lag_Close', 'Lag_Momentum', 'MACD', 'MACD_Signal',
                           'BB_Position', 'MFI', 'Stoch_K', 'Williams_R', 
                           'Volatility_Ratio', 'Trend_Strength', 'Market_Regime_Num']
        
        # Check which features are available
        available_features = [f for f in required_features if f in df.columns]
        
        if len(available_features) < 10:  # Need at least 10 features
            st.warning("⚠️ Insufficient features for ML model. Using simple signals.")
            return simple_signal_generation(df)
        
        # Get latest data
        latest_features = df[available_features].iloc[-1:].fillna(0)
        
        # 🔧 FIXED: Safe ML predictions with proper array handling
        try:
            buy_proba = st.session_state.buy_model.predict_proba(latest_features)[0]
            rr_prediction = st.session_state.rr_model.predict(latest_features)[0]
            
            # Safe confidence calculation
            if len(buy_proba) > 1:
                confidence = buy_proba[1] * 100
            else:
                confidence = (buy_proba[0] * 100) if buy_proba[0] > 0.5 else ((1 - buy_proba[0]) * 100)
            
            confidence = max(min(confidence, 85), 15)  # Conservative range
            
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
                'method': f'🛡️ Anti-Overfitting ML Model ({len(available_features)} features)',
                'buy_score': df.get('Buy_Score', pd.Series([0.5])).iloc[-1] if 'Buy_Score' in df.columns else 0.5,
                'enhanced_buy_score': df.get('Enhanced_Buy_Score', pd.Series([0.5])).iloc[-1] if 'Enhanced_Buy_Score' in df.columns else 0.5
            }
            
        except Exception as pred_error:
            st.warning(f"🛡️ ML model prediction error: {str(pred_error)}. Using fallback.")
            return simple_signal_generation(df)
        
    except Exception as e:
        st.warning(f"🛡️ ML model error: {e}. Using simple signals.")
        return simple_signal_generation(df)

def simple_signal_generation(df):
    """Enhanced fallback signal generation"""
    try:
        latest = df.iloc[-1]
        
        # Get indicators
        rsi = latest.get('RSI', 50)
        momentum = latest.get('Momentum', 0)
        price = latest.get('close', 0)
        sma_10 = latest.get('SMA_10', price)
        buy_score = latest.get('Buy_Score', 0.5)
        enhanced_buy_score = latest.get('Enhanced_Buy_Score', buy_score)
        
        # Scoring logic
        base_score = 0
        sell_score = 0
        
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
        
        # Enhanced scoring
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
            'confidence': 50,
            'buy_probability': 0.5,
            'predicted_rr': 1.0,
            'method': 'Error Fallback',
            'buy_score': 0,
            'enhanced_buy_score': 0
        }

def calculate_performance_metrics():
    """Trading performance analytics"""
    try:
        tracker = st.session_state.get('performance_tracker', [])
        
        if not tracker:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0
            }
        
        # Calculate basic metrics
        total_trades = len(tracker)
        wins = sum(1 for trade in tracker if trade.get('pnl', 0) > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(trade.get('pnl', 0) for trade in tracker)
        avg_win = np.mean([t['pnl'] for t in tracker if t.get('pnl', 0) > 0]) if wins > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in tracker if t.get('pnl', 0) < 0]) if (total_trades - wins) > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    except:
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}

def fetch_latest_candle(groww, symbol, interval_minutes=10, max_candles=50):
    """Fetch latest candle data"""
    try:
        if not groww or not hasattr(groww, 'get_instrument_by_groww_symbol'):
            st.error("Groww API not properly initialized")
            return None
            
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
        
        # Calculate start time
        days_needed = max((max_candles * interval_minutes) / (24 * 60), 5)
        days_needed = min(days_needed, 30)
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
        
        if not data or 'candles' not in data:
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

def perform_complete_analysis(groww, selected_symbol, interval_minutes, groq_key, selected_groq_model, account_balance, risk_per_trade, groq_available):
    """Complete analysis with anti-overfitting compatibility"""
    try:
        # Fetch data
        df = fetch_latest_candle(groww, selected_symbol, interval_minutes, 100)
        
        if df is None or len(df) < 20:
            return None, "Failed to fetch sufficient data"
        
        # Generate ML signal
        ml_signal = generate_ml_signal(df)
        
        # Generate Groq signal
        groq_signal = "UNKNOWN"
        if groq_available and groq_key:
            groq_signal = call_groq_llm(df, groq_key, selected_groq_model, selected_symbol)
        
        # Final signal
        if groq_signal != "UNKNOWN":
            final_signal = ml_signal.copy()
            final_signal['groq_signal'] = groq_signal
            final_signal['consensus'] = "AGREEMENT" if ml_signal['action'] == groq_signal else "MIXED"
            # Boost confidence if they agree
            if ml_signal['action'] == groq_signal and ml_signal['action'] in ['BUY', 'SELL']:
                final_signal['confidence'] = min(ml_signal['confidence'] + 10, 85)
        else:
            final_signal = ml_signal.copy()
            final_signal['groq_signal'] = "Not Available"
            final_signal['consensus'] = "ML Only"
        
        # Calculate risk levels
        current_price = df['close'].iloc[-1]
        
        # Simple risk calculation
        volatility = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 1.0
        volatility = max(volatility, 1.0)
        
        if final_signal['action'] == "BUY":
            stop_loss = current_price - (volatility * 1.2)
            take_profit_1 = current_price + (volatility * 1.5)
            take_profit_2 = current_price + (volatility * 2.5)
        elif final_signal['action'] == "SELL":
            stop_loss = current_price + (volatility * 1.2)
            take_profit_1 = current_price - (volatility * 1.5)
            take_profit_2 = current_price - (volatility * 2.5)
        else:
            stop_loss = current_price - volatility
            take_profit_1 = current_price + volatility
            take_profit_2 = current_price + (volatility * 2)
        
        risk_levels = {
            "stop_loss": round(stop_loss, 2),
            "take_profit_1": round(take_profit_1, 2),
            "take_profit_2": round(take_profit_2, 2),
            "risk_reward_ratio": round(abs(take_profit_2 - current_price) / abs(current_price - stop_loss), 2) if abs(current_price - stop_loss) > 0 else 1.0
        }
        
        # Calculate position size
        risk_amount = account_balance * (risk_per_trade / 100)
        price_risk = abs(current_price - stop_loss)
        quantity = int(risk_amount / price_risk) if price_risk > 0 else 0
        quantity = min(quantity, 1000)  # Cap at 1000 shares
        
        # Assemble analysis
        analysis_data = {
            'df': df,
            'ml_signal': ml_signal,
            'groq_signal': groq_signal,
            'final_signal': final_signal,
            'current_price': current_price,
            'risk_levels': risk_levels,
            'quantity': quantity,
            'timestamp': datetime.now(),
            'symbol': selected_symbol
        }
        
        return analysis_data, "✅ Enhanced analysis completed with anti-overfitting model!"
        
    except Exception as e:
        return None, f"Analysis failed: {str(e)}"

def display_analysis_results(analysis_data):
    """Display analysis results"""
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
        st.markdown(f"<div style='text-align:center;'>{final_signal.get('consensus', 'Single Source')}</div>", unsafe_allow_html=True)
    
    # Technical Analysis
    st.markdown("### 🔬 Technical Analysis Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    latest = df.iloc[-1]
    
    with col1:
        rsi = latest.get('RSI', 0)
        rsi_color = "🟢" if rsi < 30 else ("🔴" if rsi > 70 else "🟡")
        st.metric("RSI", f"{rsi_color} {rsi:.1f}")
        
        momentum = latest.get('Momentum', 0)
        momentum_color = "🟢" if momentum > 0 else "🔴"
        st.metric("Momentum", f"{momentum_color} {momentum:.2f}")
    
    with col2:
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_color = "🟢" if macd > macd_signal else "🔴"
        st.metric("MACD", f"{macd_color} {macd:.4f}")
        
        bb_pos = latest.get('BB_Position', 0.5)
        bb_color = "🟢" if bb_pos < 0.2 else ("🔴" if bb_pos > 0.8 else "🟡")
        st.metric("BB Position", f"{bb_color} {bb_pos:.3f}")
    
    with col3:
        mfi = latest.get('MFI', 50)
        mfi_color = "🟢" if mfi < 30 else ("🔴" if mfi > 70 else "🟡")
        st.metric("MFI", f"{mfi_color} {mfi:.1f}")
        
        stoch_k = latest.get('Stoch_K', 50)
        stoch_color = "🟢" if stoch_k < 20 else ("🔴" if stoch_k > 80 else "🟡")
        st.metric("Stochastic %K", f"{stoch_color} {stoch_k:.1f}")
    
    with col4:
        williams_r = latest.get('Williams_R', -50)
        williams_color = "🟢" if williams_r < -80 else ("🔴" if williams_r > -20 else "🟡")
        st.metric("Williams %R", f"{williams_color} {williams_r:.1f}")
        
        buy_score = latest.get('Buy_Score', 0)
        score_color = "🟢" if buy_score > 0.8 else ("🟡" if buy_score > 0.6 else "🔴")
        st.metric("Buy Score", f"{score_color} {buy_score:.3f}")
    
    # Enhanced Buy Score display
    enhanced_buy_score = latest.get('Enhanced_Buy_Score', buy_score)
    enhanced_color = "🟢" if enhanced_buy_score > 0.8 else ("🟡" if enhanced_buy_score > 0.6 else "🔴")
    st.metric("🆕 Enhanced Buy Score", f"{enhanced_color} {enhanced_buy_score:.3f}")
    
    # Trade Analysis
    if final_signal['action'] in ["BUY", "SELL"]:
        st.markdown("---")
        st.markdown("## 📈 Trade Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Current Price", f"₹{current_price:.2f}")
            st.metric("📦 Quantity", f"{quantity:,} shares")
        
        with col2:
            investment = current_price * quantity
            st.metric("💵 Investment", f"₹{investment:,.0f}")
            st.metric("🛡️ Stop Loss", f"₹{risk_levels['stop_loss']}")
        
        with col3:
            st.metric("🎯 Take Profit 1", f"₹{risk_levels['take_profit_1']}")
            st.metric("🎯 Take Profit 2", f"₹{risk_levels['take_profit_2']}")
        
        with col4:
            st.metric("⚖️ Risk:Reward", f"1:{risk_levels['risk_reward_ratio']}")
            max_profit = abs(risk_levels['take_profit_2'] - current_price) * quantity
            max_loss = abs(current_price - risk_levels['stop_loss']) * quantity
            expected_profit = (max_profit * 0.65) - (max_loss * 0.35)  # Conservative estimate
            st.metric("📊 Expected Profit", f"₹{expected_profit:,.0f}")
    
    # Recent data
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
    
    if not st.session_state.get('models_loaded', False):
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

# Sidebar Authentication
st.sidebar.title("🔐 API Authentication")
grow_key = st.sidebar.text_input("Groww API token", type="password", key="grow_api_key")
groq_key = st.sidebar.text_input("Groq API key", type="password", key="groq_api_key")

# System initialization
if not grow_key:
    st.warning("Please enter your Groww API token in the sidebar.")
    st.stop()

# 🔧 FIXED: Initialize Groww API safely
with st.spinner("Initializing Groww API safely..."):
    groww, instruments_df, init_message = initialize_groww_safely()

if groww is None:
    st.error(f"❌ {init_message}")
    st.info("💡 Try refreshing the page or check your API key")
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
if instruments_df is not None:
    symbols_list = instruments_df["groww_symbol"].sort_values().unique().tolist()
    default_symbol = "NSE-NIFTY" if "NSE-NIFTY" in symbols_list else symbols_list[0]
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols_list, index=symbols_list.index(default_symbol))
else:
    selected_symbol = st.sidebar.text_input("Enter Symbol", value="NSE-NIFTY")

# Settings
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Account Settings")
account_balance = st.sidebar.number_input("Account Balance (₹)", value=100000, min_value=10000, step=10000)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

st.sidebar.subheader("📊 Analysis Settings")
interval_minutes = st.sidebar.selectbox("Candle Interval", [5, 10, 15, 30, 60], index=1)
if groq_available:
    selected_groq_model = st.sidebar.selectbox("Groq Model", groq_models, index=0)
else:
    selected_groq_model = None

# Auto-refresh settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto Refresh")
auto_refresh_enabled = st.sidebar.checkbox("Enable Auto Refresh", value=st.session_state.get('auto_refresh', False))
if auto_refresh_enabled:
    refresh_options = {"1 minute": 60, "2 minutes": 120, "5 minutes": 300, "10 minutes": 600}
    refresh_interval = st.sidebar.selectbox("Refresh Interval", list(refresh_options.keys()), index=1)
    refresh_seconds = refresh_options[refresh_interval]
    st.session_state.auto_refresh = True
else:
    st.session_state.auto_refresh = False

# Main Analysis Buttons
st.markdown("## 🎯 Anti-Overfitting Trading Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    analyze_clicked = st.button("🛡️ Analyze Symbol", type="primary", use_container_width=True)

with col2:
    refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True)

with col3:
    if st.button("🧹 Clear Results", use_container_width=True):
        st.session_state.analysis_data = None
        st.session_state.last_refresh = None
        st.rerun()

# Auto-refresh logic
if st.session_state.get('auto_refresh', False) and st.session_state.get('last_refresh') and st.session_state.get('analysis_data'):
    time_diff = (datetime.now() - st.session_state.last_refresh).seconds
    if time_diff >= refresh_seconds:
        st.info(f"🔄 Auto-refreshing after {refresh_interval}...")
        refresh_clicked = True

# Perform analysis
if analyze_clicked or refresh_clicked:
    with st.spinner(f"Analyzing {selected_symbol} with anti-overfitting model..."):
        
        analysis_data, message = perform_complete_analysis(
            groww, selected_symbol, interval_minutes, 
            groq_key, selected_groq_model,
            account_balance, risk_per_trade, groq_available
        )
        
        if analysis_data is not None:
            st.session_state.analysis_data = analysis_data
            st.session_state.last_refresh = datetime.now()
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")

# Display results if available
if st.session_state.get('analysis_data'):
    display_analysis_results(st.session_state.analysis_data)
    
    # Action buttons for trading decisions
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
            if st.button("💾 Save Signal"):
                # Save signal data
                signal_data = {
                    'timestamp': datetime.now(),
                    'symbol': selected_symbol,
                    'action': final_signal['action'],
                    'confidence': final_signal['confidence'],
                    'method': final_signal.get('method', 'Anti-Overfitting ML'),
                    'price': st.session_state.analysis_data['current_price'],
                    'pnl': 0  # Will be updated when position is closed
                }
                
                # Add to performance tracker
                if 'performance_tracker' not in st.session_state:
                    st.session_state.performance_tracker = []
                st.session_state.performance_tracker.append(signal_data)
                
                st.success(f"💾 Signal saved! Total signals: {len(st.session_state.performance_tracker)}")
        
        with action_col3:
            if st.button("📊 Analyze Another"):
                st.session_state.analysis_data = None
                st.session_state.last_refresh = None
                st.rerun()

# System status dashboard
st.markdown("---")
st.markdown("### 🛡️ Anti-Overfitting System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    models_status = "✅ Anti-Overfitting" if st.session_state.get('models_loaded', False) else "❌ Not Loaded"
    st.metric("🛡️ ML Models", models_status)

with status_col2:
    groq_status = "✅ Connected" if groq_available else "❌ Disconnected"
    st.metric("🧠 Groq LLM", groq_status)

with status_col3:
    features_count = 16
    st.metric("📊 Safe Features", f"✅ {features_count}")

with status_col4:
    system_status = "🟢 REALISTIC MODE" if (st.session_state.get('models_loaded', False) and groq_available) else "🟡 PARTIAL OPERATION"
    st.metric("🎯 System Status", system_status)

# Enhanced footer with performance metrics
st.markdown("---")
footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

with footer_col1:
    if st.session_state.get('last_refresh'):
        st.caption(f"🕐 Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    else:
        st.caption("🕐 No analysis data")

with footer_col2:
    st.caption(f"📊 Symbol: {selected_symbol}")
    portfolio_count = len(st.session_state.get('portfolio', {}))
    st.caption(f"💼 Portfolio: {portfolio_count} symbols")

with footer_col3:
    if st.session_state.get('auto_refresh', False):
        st.caption("🔄 Auto-refresh ON")
    else:
        st.caption("🔄 Auto-refresh OFF")
    
    st.caption("🛡️ Anti-overfitting: ON")

with footer_col4:
    st.caption("🎯 Realistic Performance Mode")
    
    performance = calculate_performance_metrics()
    total_trades = performance.get('total_trades', 0)
    win_rate = performance.get('win_rate', 0)
    if total_trades > 0:
        st.caption(f"📈 Performance: {win_rate:.1f}% win rate ({total_trades} trades)")
    else:
        st.caption("📈 Performance: No trades yet")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
<b>🛡️ Anti-Overfitting Trading Signal System v3.0</b><br>
Realistic ML Predictions + Groq LLM Intelligence + Conservative Risk Management<br>
🎯 Target Accuracy: 60-75% | 🚫 No Data Leakage | 🛡️ Anti-Overfitting Protection<br>
Built for <b>Real Trading Success</b> with Honest Performance Expectations
</div>
""", unsafe_allow_html=True)

# Risk disclaimer
st.error("""
⚠️ **ANTI-OVERFITTING TRADING SYSTEM DISCLAIMER** ⚠️

🛡️ **This system uses anti-overfitting models with realistic 60-75% accuracy expectations.**

💰 **Trading involves substantial risk of loss and is not suitable for all investors.**

📊 **Anti-overfitting models provide more realistic but not infallible predictions.**

🎯 **Expected performance: 60-75% accuracy (not 99%+ which indicates overfitting).**

🧠 **Always conduct your own analysis and consider consulting a qualified financial advisor.**

🔍 **Paper trade first to validate performance before risking real capital.**

⚡ **The conservative approach reduces false confidence but doesn't eliminate trading risks.**

📱 **Use proper position sizing, stop losses, and risk management at all times.**

🛡️ **Better to have realistic 65% accuracy than fake 99% that fails in live trading.**

**By using this anti-overfitting system, you acknowledge these realistic performance expectations.**
""")
