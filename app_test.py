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

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Enhanced Trading Signal Tool",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.buy_model = None
    st.session_state.rr_model = None
    st.session_state.last_refresh = None
    st.session_state.analysis_data = None
    st.session_state.auto_refresh = False

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
            return True, "Models loaded successfully"
        else:
            return False, "Model files not found. Please run retraining.py first."
    except Exception as e:
        return False, f"Error loading models: {str(e)}"

def initialize_groww_safely():
    """Initialize Groww API safely"""
    try:
        from growwapi import GrowwAPI
        grow_key = st.session_state.get("grow_api_key", "")
        
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
            return groww, instruments_df, "Groww API initialized successfully"
            
        except FileNotFoundError:
            return None, None, "instruments.csv file not found"
        except Exception as e:
            return None, None, f"Error loading instruments: {str(e)}"
            
    except ImportError:
        return None, None, "GrowwAPI not installed. Please install: pip install groww-api"
    except Exception as e:
        return None, None, f"Error initializing Groww API: {str(e)}"

def get_groq_models(groq_key):
    """Get available Groq models"""
    try:
        import groq
        from groq import Groq
        client = Groq(api_key=groq_key)
        models = client.models.list()
        return [m.id for m in models.data], None
    except ImportError:
        return [], "Groq Python lib not installed! Install: pip install groq"
    except Exception as e:
        return [], f"Groq API error: {e}"

def call_groq_llm(df, groq_key, model_name, symbol):
    """Get trading signal from Groq LLM"""
    try:
        import groq
        from groq import Groq
        
        # Prepare recent candle data
        recent_data = df.tail(5)
        candles_info = []
        
        for _, row in recent_data.iterrows():
            candle_info = f"Time: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}, "
            candle_info += f"Open: {row['open']:.2f}, High: {row['high']:.2f}, "
            candle_info += f"Low: {row['low']:.2f}, Close: {row['close']:.2f}, "
            candle_info += f"Volume: {row['volume']:,.0f}"
            candles_info.append(candle_info)
        
        prompt = f"""
You are an expert trading analyst. Analyze this recent candlestick data for {symbol}:

{chr(10).join(candles_info)}

Based on ONLY this price action and volume data, what is your trading recommendation for the NEXT candle?

Rules:
- Respond with exactly ONE word: BUY, SELL, or HOLD
- Consider: price momentum, volume patterns, support/resistance levels
- BUY: if you see bullish momentum, volume confirmation, or bounce from support
- SELL: if you see bearish momentum, volume selling, or rejection from resistance  
- HOLD: if sideways/unclear pattern

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
            return "HOLD"  # Default to HOLD if unclear response
            
    except Exception as e:
        st.warning(f"Groq LLM error: {e}")
        return "UNKNOWN"

def compute_technical_indicators(df):
    """Add comprehensive technical indicators including Buy_Score"""
    try:
        df = df.copy()
        
        # Moving averages
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
        
        # Volume indicators
        df['Volume_MA'] = df['volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['volume'] / (df['Volume_MA'] + 1)
        
        # **NEW: Calculate Buy_Score (same logic as training)**
        cond1 = (df["RSI"] < 45) & (df["Momentum"] > 0)
        cond2 = (df["close"] > df["SMA_10"]) & (df["Volume_Ratio"] > 1.1)
        cond3 = (df["MACD"] > df["MACD_Signal"]) & ((df["MACD"] - df["MACD_Signal"]) > 0)
        cond4 = (df["Momentum"] > df["Momentum"].quantile(0.6)) & (df["RSI"] < 70)
        
        # Weighted scoring (same as training)
        df['Buy_Score'] = (cond1.astype(int) * 0.3) + (cond2.astype(int) * 0.2) + (cond3.astype(int) * 0.3) + (cond4.astype(int) * 0.2)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return df

def generate_ml_signal(df):
    """Generate ML-based trading signal with enhanced features"""
    try:
        if not st.session_state.models_loaded:
            st.info("🔄 Models not loaded. Using simple signal generation.")
            return simple_signal_generation(df)
        
        # **UPDATED: Include Buy_Score in features**
        all_features = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility", 
                       "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal", "Buy_Score"]
        core_features = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility", 
                        "MACD", "MACD_Signal"]  # Minimum required
        
        available_features = [f for f in all_features if f in df.columns]
        core_available = [f for f in core_features if f in df.columns]
        
        # Check if we have minimum required features
        if len(core_available) < 5:
            st.warning("⚠️ Insufficient technical indicators. Using simple signals.")
            return simple_signal_generation(df)
        
        # Use available features
        latest_features = df[available_features].iloc[-1:].fillna(0)
        
        # Get ML predictions
        buy_proba = st.session_state.buy_model.predict_proba(latest_features)[0]
        rr_prediction = st.session_state.rr_model.predict(latest_features)
        
        confidence = buy_proba[1] * 100
        
        # **ENHANCED: Adjust confidence based on Buy_Score if available**
        buy_score = None
        if 'Buy_Score' in available_features:
            buy_score = latest_features['Buy_Score'].iloc[0]
            # Boost/reduce confidence based on buy score
            score_boost = (buy_score - 0.5) * 20  # Boost up to ±10%
            confidence = max(min(confidence + score_boost, 95), 5)
        
        # Determine action
        if confidence >= 65:
            action = "BUY"
        elif confidence <= 35:
            action = "SELL"
        else:
            action = "HOLD"
        
        return {
            'action': action,
            'confidence': confidence,
            'buy_probability': buy_proba[1],
            'predicted_rr': max(rr_prediction[0], 0.01),
            'method': f'ML Model ({len(available_features)} features)',
            'buy_score': buy_score
        }
        
    except Exception as e:
        st.warning(f"ML model error: {e}. Using simple signals.")
        return simple_signal_generation(df)

def simple_signal_generation(df):
    """Fallback simple signal generation"""
    try:
        latest = df.iloc[-1]
        
        rsi = latest.get('RSI', 50)
        momentum = latest.get('Momentum', 0)
        price = latest.get('close', 0)
        sma_10 = latest.get('SMA_10', price)
        buy_score = latest.get('Buy_Score', 0.5)
        
        # Simple scoring with Buy_Score integration
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
        
        # Integrate Buy_Score
        score_boost = buy_score * 3  # Up to 3 extra points
        final_buy_score = base_score + score_boost
        
        if final_buy_score >= 4:
            action = "BUY"
            confidence = min(final_buy_score * 12, 85)
        elif sell_score >= 3:
            action = "SELL"
            confidence = min(sell_score * 20, 85)
        else:
            action = "HOLD"
            confidence = 50
        
        return {
            'action': action,
            'confidence': confidence,
            'buy_probability': confidence / 100,
            'predicted_rr': 1.5,
            'method': 'Simple Rules',
            'buy_score': buy_score
        }
        
    except Exception as e:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'buy_probability': 0.5,
            'predicted_rr': 1.0,
            'method': 'Error Fallback',
            'buy_score': 0
        }

def combine_ml_and_groq_signals(ml_signal, groq_signal):
    """Combine ML and Groq signals for enhanced accuracy"""
    base_action = ml_signal['action']
    base_confidence = ml_signal['confidence']
    
    # Agreement boosts confidence
    if base_action == groq_signal and base_action in ['BUY', 'SELL']:
        enhanced_confidence = min(base_confidence + 15, 95)
        signal_strength = "STRONG"
        consensus = "AGREEMENT"
    
    # Disagreement reduces confidence
    elif base_action != groq_signal and base_action in ['BUY', 'SELL']:
        enhanced_confidence = max(base_confidence - 10, 25)
        signal_strength = "WEAK"
        consensus = "MIXED"
    
    else:
        enhanced_confidence = base_confidence
        signal_strength = "MODERATE"
        consensus = "NEUTRAL"
    
    return {
        'action': base_action,  # Always use ML model decision
        'confidence': enhanced_confidence,
        'signal_strength': signal_strength,
        'consensus': consensus,
        'ml_signal': base_action,
        'groq_signal': groq_signal,
        'agreement': base_action == groq_signal,
        'original_confidence': base_confidence,
        'buy_score': ml_signal.get('buy_score', 0)
    }

def calculate_position_size(current_price, stop_loss, account_balance, risk_per_trade):
    """Calculate optimal position size"""
    try:
        if stop_loss == 0 or current_price == stop_loss:
            return 0
        
        risk_amount = account_balance * (risk_per_trade / 100)
        price_risk = abs(current_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = int(risk_amount / price_risk)
        max_shares_by_value = int((account_balance * 0.10) / current_price)  # Max 10% of account
        
        return min(position_size, max_shares_by_value, 1000)  # Cap at 1000 shares
        
    except:
        return 0

def calculate_risk_levels(df, action, current_price):
    """Calculate stop loss, take profit levels, and risk/reward ratio"""
    latest = df.iloc[-1]
    atr = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 1.0
    atr = max(atr, 1.0)
    buffer = atr * 1.2

    if action == "BUY":
        stop_loss = round(current_price - buffer, 2)
        take_profit_1 = round(current_price + buffer * 1.5, 2)
        take_profit_2 = round(current_price + buffer * 2.5, 2)
    elif action == "SELL":
        stop_loss = round(current_price + buffer, 2)
        take_profit_1 = round(current_price - buffer * 1.5, 2)
        take_profit_2 = round(current_price - buffer * 2.5, 2)
    else:
        stop_loss = round(current_price - buffer, 2)
        take_profit_1 = round(current_price + buffer, 2)
        take_profit_2 = round(current_price + buffer * 2, 2)

    price_risk = abs(current_price - stop_loss)
    price_reward = abs(take_profit_2 - current_price)
    risk_reward_ratio = round(price_reward / price_risk, 2) if price_risk > 0 else 1.0

    return {
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "risk_reward_ratio": risk_reward_ratio,
        "atr": atr
    }

def fetch_latest_candle(groww, symbol, interval_minutes=10, max_candles=50):
    """Fetch latest candle data with robust error handling"""
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
        
        # Add technical indicators (including Buy_Score)
        df = compute_technical_indicators(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching  {str(e)}")
        return None

def perform_complete_analysis(groww, selected_symbol, interval_minutes, groq_key, selected_groq_model, account_balance, risk_per_trade, groq_available):
    """Perform complete trading analysis and return results"""
    try:
        # Fetch data
        df = fetch_latest_candle(groww, selected_symbol, interval_minutes, 100)
        
        if df is None or len(df) < 20:
            return None, "Failed to fetch sufficient data"
        
        # Generate ML signal
        ml_signal = generate_ml_signal(df)
        
        # Generate Groq signal if available
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
        
        # Calculate risk levels and position size
        current_price = df['close'].iloc[-1]
        risk_levels = calculate_risk_levels(df, final_signal['action'], current_price)
        quantity = calculate_position_size(current_price, risk_levels['stop_loss'], account_balance, risk_per_trade)
        
        # Compile complete analysis
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
        
        return analysis_data, "Analysis completed successfully"
        
    except Exception as e:
        return None, f"Analysis failed: {str(e)}"

def display_analysis_results(analysis_data):
    """Display complete analysis results with Buy_Score"""
    df = analysis_data['df']
    ml_signal = analysis_data['ml_signal']
    groq_signal = analysis_data['groq_signal']
    final_signal = analysis_data['final_signal']
    current_price = analysis_data['current_price']
    risk_levels = analysis_data['risk_levels']
    quantity = analysis_data['quantity']
    
    # Display Results
    st.markdown("## 📈 Trading Signal Analysis")
    
    # Signal overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🤖 ML Model Signal")
        ml_color = "#22a218" if ml_signal['action'] == "BUY" else ("#d32f2f" if ml_signal['action'] == "SELL" else "#f39c12")
        ml_emoji = "🟢" if ml_signal['action'] == "BUY" else ("🔴" if ml_signal['action'] == "SELL" else "🟡")
        st.markdown(f"<div style='font-size:2em; text-align:center; color:{ml_color};'>{ml_emoji} {ml_signal['action']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'>Confidence: <b>{ml_signal['confidence']:.1f}%</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;'>Method: {ml_signal['method']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🧠 Groq LLM Signal")
        if groq_signal != "UNKNOWN":
            groq_color = "#22a218" if groq_signal == "BUY" else ("#d32f2f" if groq_signal == "SELL" else "#f39c12")
            groq_emoji = "🟢" if groq_signal == "BUY" else ("🔴" if groq_signal == "SELL" else "🟡")
            st.markdown(f"<div style='font-size:2em; text-align:center; color:{groq_color};'>{groq_emoji} {groq_signal}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;'>Pattern Analysis</div>", unsafe_allow_html=True)
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
    
    # **NEW: Technical Analysis with Buy_Score**
    st.markdown("### 📊 Technical Analysis")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest = df.iloc[-1]
    
    with col1:
        current_rsi = latest.get('RSI', 0)
        st.metric("RSI", f"{current_rsi:.1f}")
    
    with col2:
        current_momentum = latest.get('Momentum', 0)
        st.metric("Momentum", f"{current_momentum:.2f}")
    
    with col3:
        current_macd = latest.get('MACD', 0)
        st.metric("MACD", f"{current_macd:.4f}")
    
    with col4:
        current_volume_ratio = latest.get('Volume_Ratio', 1)
        st.metric("Volume Ratio", f"{current_volume_ratio:.2f}x")
    
    with col5:
        # **NEW: Display Buy_Score**
        current_buy_score = latest.get('Buy_Score', 0)
        score_color = "🟢" if current_buy_score > 0.6 else ("🟡" if current_buy_score > 0.4 else "🔴")
        st.metric("Buy Score", f"{score_color} {current_buy_score:.2f}")
    
    # Trade Analysis (existing code continues...)
    if final_signal['action'] in ["BUY", "SELL"]:
        investment_amount = current_price * quantity
        
        st.markdown("---")
        st.markdown("## 📈 Trade Analysis")
        
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
            expected_profit = (max_profit * 0.6) - (max_loss * 0.4)  # 60% win rate assumption
            st.metric("📊 Expected Profit", f"₹{expected_profit:,.0f}")
    
    # Agreement Analysis (if Groq available)
    if groq_signal != "UNKNOWN":
        st.markdown("### 🤝 Signal Agreement Analysis")
        agreement_col1, agreement_col2 = st.columns(2)
        
        with agreement_col1:
            st.write(f"**ML Model says:** {ml_signal['action']} ({ml_signal['confidence']:.1f}%)")
            st.write(f"**Groq LLM says:** {groq_signal}")
            st.write(f"**Agreement:** {'✅ YES' if final_signal.get('agreement') else '❌ NO'}")
        
        with agreement_col2:
            st.write(f"**Original ML Confidence:** {final_signal.get('original_confidence', 0):.1f}%")
            st.write(f"**Enhanced Confidence:** {final_signal['confidence']:.1f}%")
            confidence_change = final_signal['confidence'] - final_signal.get('original_confidence', 0)
            st.write(f"**Confidence Change:** {confidence_change:+.1f}%")
    
    # Recent market data
    st.markdown("### 📋 Recent Market Data")
    display_df = df.tail(10)[['timestamp', 'close', 'open', 'high', 'low', 'volume', 'RSI', 'Buy_Score']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_df = display_df.round(3)
    st.dataframe(display_df, use_container_width=True)

# Main App Layout
st.title("🚀 Enhanced Trading Signal Tool")
st.markdown("### ML Model + Groq LLM Analysis with Weighted Buy Score")

# **NEW: Model Information in Sidebar**
with st.sidebar:
    st.markdown("### 🤖 Model Information")
    
    if not st.session_state.models_loaded:
        success, message = load_models_safely()
        if success:
            st.success("✅ Models Loaded")
        else:
            st.error("❌ Models Not Loaded")
            if st.button("🔄 Reload Models"):
                success, message = load_models_safely()
                if success:
                    st.rerun()
    else:
        st.success("✅ Models Loaded")
        
        # Try to show feature importance
        try:
            if hasattr(st.session_state.buy_model, 'feature_importances_'):
                st.markdown("**Top Features:**")
                features = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility", 
                           "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal", "Buy_Score"]
                importances = st.session_state.buy_model.feature_importances_
                
                # Show top 5 features
                feature_importance = list(zip(features[:len(importances)], importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                for feat, imp in feature_importance[:5]:
                    st.text(f"{feat}: {imp:.3f}")
        except:
            pass

# Sidebar Authentication
st.sidebar.title("🔐 API Authentication")
grow_key = st.sidebar.text_input("Groww API token", type="password", key="grow_api_key")
groq_key = st.sidebar.text_input("Groq API key (optional)", type="password", key="groq_api_key")

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
st.markdown("## 🎯 Trading Analysis")

# Action buttons row
button_col1, button_col2, button_col3, button_col4 = st.columns(4)

with button_col1:
    analyze_clicked = st.button("🔍 Analyze Symbol", type="primary", use_container_width=True)

with button_col2:
    refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True)

with button_col3:
    if st.session_state.analysis_:
        quick_refresh = st.button("⚡ Quick Update", use_container_width=True)
    else:
        quick_refresh = False

with button_col4:
    if st.button("🧹 Clear Results", use_container_width=True):
        st.session_state.analysis_data = None
        st.session_state.last_refresh = None
        st.rerun()

# Auto-refresh logic
if st.session_state.auto_refresh and st.session_state.analysis_:
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
    with st.spinner(f"{'Refreshing' if refresh_clicked or quick_refresh else 'Analyzing'} {selected_symbol}..."):
        
        # Perform complete analysis
        analysis_data, message = perform_complete_analysis(
            groww, selected_symbol, interval_minutes, 
            groq_key, selected_groq_model if groq_available else None,
            account_balance, risk_per_trade, groq_available
        )
        
        if analysis_data:
            st.session_state.analysis_data = analysis_data
            st.session_state.last_refresh = datetime.now()
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")

# Display results if available
if st.session_state.analysis_:
    display_analysis_results(st.session_state.analysis_data)
    
    # Action buttons
    final_signal = st.session_state.analysis_data['final_signal']
    if final_signal['action'] in ["BUY", "SELL"]:
        st.markdown("---")
        st.markdown("### 🎬 Trading Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button(f"📈 Note: {final_signal['action']} Signal", type="primary"):
                st.success(f"✅ {final_signal['action']} signal noted! Execute manually in your trading app.")
        
        with action_col2:
            if st.button("📋 Save Signal"):
                current_price = st.session_state.analysis_data['current_price']
                quantity = st.session_state.analysis_data['quantity']
                risk_levels = st.session_state.analysis_data['risk_levels']
                groq_signal = st.session_state.analysis_data['groq_signal']
                ml_signal = st.session_state.analysis_data['ml_signal']
                
                signal_data = {
                    'timestamp': datetime.now(),
                    'symbol': selected_symbol,
                    'ml_action': ml_signal['action'],
                    'groq_action': groq_signal,
                    'final_action': final_signal['action'],
                    'confidence': final_signal['confidence'],
                    'buy_score': final_signal.get('buy_score', 0),
                    'price': current_price,
                    'quantity': quantity if final_signal['action'] in ["BUY", "SELL"] else 0,
                    'stop_loss': risk_levels['stop_loss'] if final_signal['action'] in ["BUY", "SELL"] else 0,
                    'take_profit': risk_levels['take_profit_2'] if final_signal['action'] in ["BUY", "SELL"] else 0
                }
                
                signal_df = pd.DataFrame([signal_data])
                filename = f"signals_{selected_symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                signal_df.to_csv(filename, index=False)
                st.success(f"📁 Signal saved to {filename}")
        
        with action_col3:
            if st.button("📊 Analyze Another Symbol"):
                st.rerun()

# Auto-refresh display
if st.session_state.auto_refresh and st.session_state.analysis_:
    st.info(f"🔄 Auto-refresh enabled - updating every {refresh_interval}")
    if st.button("⏹️ Stop Auto-refresh"):
        st.session_state.auto_refresh = False
        st.rerun()
    
    # Auto-refresh timer
    time.sleep(1)
    st.rerun()

# Help section
with st.expander("📖 How to Use This Tool"):
    st.markdown("""
    **🔍 Analyze Symbol**: Run complete analysis with ML + Groq signals
    
    **🔄 Refresh Data**: Get latest market data and updated signals
    
    **⚡ Quick Update**: Fast refresh of current analysis
    
    **🧹 Clear Results**: Remove current analysis from screen
    
    **Auto Refresh**: Automatically update data at set intervals
    
    **Signal Confidence**:
    - **85%+**: Very Strong Signal
    - **70-85%**: Strong Signal  
    - **50-70%**: Moderate Signal
    - **Below 50%**: Weak Signal
    
    **Buy Score**:
    - **0.8+**: 🟢 Very Strong Buy Conditions
    - **0.6-0.8**: 🟡 Moderate Buy Conditions
    - **Below 0.6**: 🔴 Weak Buy Conditions
    
    **Agreement Status**:
    - **AGREEMENT**: ML and Groq both suggest same action
    - **MIXED**: Different recommendations
    - **ML ONLY**: Only ML model available
    """)

# Risk disclaimer
st.markdown("---")
st.error("""
⚠️ **IMPORTANT RISK DISCLAIMER** ⚠️

🚨 This tool is for educational and research purposes only.
💰 Trading involves substantial risk of loss.
📊 Past performance does not guarantee future results.
🧠 Always conduct your own analysis and consider consulting a financial advisor.
🎯 Never invest more than you can afford to lose.
""")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if st.session_state.last_refresh:
        st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    else:
        st.caption("No analysis data")

with footer_col2:
    st.caption(f"Symbol: {selected_symbol}")

with footer_col3:
    if st.session_state.auto_refresh:
        st.caption("🔄 Auto-refresh ON")
    else:
        st.caption("🔄 Auto-refresh OFF")
