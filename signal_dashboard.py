import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time
import traceback
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

# Initialize session state (same as app.py)
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.buy_model = None
    st.session_state.rr_model = None
    st.session_state.last_refresh = None
    st.session_state.analysis_data = None
    st.session_state.auto_refresh = False

def load_models_safely():
    """Load models exactly like app.py"""
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
    """Initialize Groww API exactly like app.py"""
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
    """Get Groq models exactly like app.py"""
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
    """Call Groq LLM exactly like app.py"""
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
    """Add technical indicators exactly like app.py"""
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
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return df

def calculate_position_size(current_price, stop_loss, account_balance, risk_per_trade):
    """FIXED: Calculate position size with proper validation and debugging"""
    try:
        # Create debug expander
        with st.expander("🔍 Position Sizing Debug"):
            st.write(f"**Account Balance:** ₹{account_balance:,}")
            st.write(f"**Risk per Trade:** {risk_per_trade}%")
            st.write(f"**Current Price:** ₹{current_price:.2f}")
            st.write(f"**Stop Loss:** ₹{stop_loss:.2f}")
            
            # Calculate risk amounts
            risk_amount = account_balance * (risk_per_trade / 100)
            price_risk = abs(current_price - stop_loss)
            
            st.write(f"**Risk Amount:** ₹{risk_amount:.2f}")
            st.write(f"**Price Risk per Share:** ₹{price_risk:.2f}")
            
            if price_risk <= 0.10:  # Less than 10 paisa
                st.warning("⚠️ Stop loss very close to current price!")
                # Use minimum meaningful risk
                price_risk = max(price_risk, current_price * 0.005)  # At least 0.5% risk
                st.write(f"**Adjusted Price Risk:** ₹{price_risk:.2f}")
            
            # Calculate basic position size
            basic_quantity = int(risk_amount / price_risk)
            st.write(f"**Basic Quantity (Risk-based):** {basic_quantity:,} shares")
            
            # Apply limits - FIXED: More reasonable limits
            max_shares_by_value = int((account_balance * 0.30) / current_price)  # Max 30% of account
            max_shares_absolute = 50 if current_price > 1000 else 1000  # Reasonable caps
            
            st.write(f"**Max by Account (30%):** {max_shares_by_value:,} shares")
            st.write(f"**Absolute Cap:** {max_shares_absolute:,} shares")
            
            quantity = min(basic_quantity, max_shares_by_value, max_shares_absolute)
            
            # FIXED: Ensure minimum quantity for small accounts
            if quantity == 0 and account_balance >= current_price and basic_quantity > 0:
                quantity = 1  # At least 1 share if account can afford it
                st.info("✅ Applied minimum quantity of 1 share")
            
            investment_required = quantity * current_price
            st.write(f"**Final Quantity:** {quantity:,} shares")
            st.write(f"**Investment Required:** ₹{investment_required:,.2f}")
            
            if investment_required > account_balance:
                st.error("❌ Investment exceeds account balance!")
                quantity = int(account_balance / current_price)
                st.write(f"**Adjusted Quantity:** {quantity:,} shares")
        
        return max(quantity, 0)  # Ensure non-negative
        
    except Exception as e:
        st.error(f"Error in position sizing: {e}")
        return 0

def calculate_risk_levels(df, action, current_price):
    """FIXED: Calculate risk levels with better ATR handling"""
    try:
        # Calculate ATR for volatility-based stops
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['TR'].rolling(14).mean().iloc[-1]
        
        # FIXED: Better fallback ATR calculation
        if pd.isna(atr) or atr == 0:
            # Use price volatility as fallback
            price_std = df['close'].rolling(14).std().iloc[-1]
            atr = price_std if not pd.isna(price_std) else current_price * 0.02
        
        # FIXED: More reasonable multipliers for different price levels
        if current_price > 20000:  # High-priced stocks like NIFTY
            atr_multiplier = 1.0  # Smaller multiplier for high-priced stocks
        elif current_price > 1000:
            atr_multiplier = 1.5
        else:
            atr_multiplier = 2.0
        
        if action == "BUY":
            stop_loss = round(current_price - (atr_multiplier * atr), 2)
            tp1 = round(current_price + (atr_multiplier * atr), 2)
            tp2 = round(current_price + (2 * atr_multiplier * atr), 2)
        else:  # SELL
            stop_loss = round(current_price + (atr_multiplier * atr), 2)
            tp1 = round(current_price - (atr_multiplier * atr), 2)
            tp2 = round(current_price - (2 * atr_multiplier * atr), 2)
        
        # FIXED: Ensure meaningful risk levels
        min_risk = current_price * 0.005  # At least 0.5% risk
        
        if action == "BUY":
            if (current_price - stop_loss) < min_risk:
                stop_loss = current_price - min_risk
                tp1 = current_price + min_risk
                tp2 = current_price + (2 * min_risk)
        else:
            if (stop_loss - current_price) < min_risk:
                stop_loss = current_price + min_risk
                tp1 = current_price - min_risk
                tp2 = current_price - (2 * min_risk)
        
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(tp2 - current_price)
        rr_ratio = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit_1': round(tp1, 2),
            'take_profit_2': round(tp2, 2),
            'risk_reward_ratio': max(rr_ratio, 0.1),  # Ensure positive RR
            'atr': round(atr, 2)
        }
        
    except Exception as e:
        st.error(f"Error calculating risk levels: {e}")
        # FIXED: Better fallback values
        fallback_risk = current_price * 0.02  # 2% risk
        if action == "BUY":
            return {
                'stop_loss': round(current_price - fallback_risk, 2),
                'take_profit_1': round(current_price + fallback_risk, 2),
                'take_profit_2': round(current_price + (2 * fallback_risk), 2),
                'risk_reward_ratio': 2.0,
                'atr': round(fallback_risk, 2)
            }
        else:
            return {
                'stop_loss': round(current_price + fallback_risk, 2),
                'take_profit_1': round(current_price - fallback_risk, 2),
                'take_profit_2': round(current_price - (2 * fallback_risk), 2),
                'risk_reward_ratio': 2.0,
                'atr': round(fallback_risk, 2)
            }

def generate_ml_signal(df):
    """Generate ML signal with debugging"""
    try:
        st.write("🔍 **DEBUG: ML Signal Generation**")
        st.write(f"Models loaded: {st.session_state.models_loaded}")
        st.write(f"DataFrame shape: {df.shape}")
        
        if not st.session_state.models_loaded:
            st.warning("⚠️ ML models not loaded - using simple signals")
            return simple_signal_generation(df)
        
        # Prepare features
        features = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility",
                   "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal"]
        
        # Check missing features
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(f"❌ Missing features: {missing_features}")
            return simple_signal_generation(df)
        
        latest_features = df[features].iloc[-1:].fillna(0)
        
        # **DEBUG: Print the exact features being passed to model**
        with st.expander("📊 Model Input Features"):
            st.dataframe(latest_features)
        
        # Get ML predictions
        buy_proba = st.session_state.buy_model.predict_proba(latest_features)[0]
        rr_prediction = st.session_state.rr_model.predict(latest_features)
        
        # **DEBUG: Print model outputs**
        with st.expander("🤖 Model Predictions"):
            st.write(f"**Buy probabilities:** [No Buy: {buy_proba[0]:.4f}, Buy: {buy_proba[1]:.4f}]")
            st.write(f"**Risk-Reward prediction:** {rr_prediction:.4f}")
        
        confidence = buy_proba[1] * 100
        st.write(f"**Calculated confidence:** {confidence:.2f}%")
        
        # Determine action based on confidence
        if confidence >= 65:
            action = "BUY"
            st.success(f"✅ Action: {action} (confidence >= 65%)")
        elif confidence <= 35:
            action = "SELL"
            st.error(f"❌ Action: {action} (confidence <= 35%)")
        else:
            action = "HOLD"
            st.info(f"⏸️ Action: {action} (35% < confidence < 65%)")
        
        return {
            'action': action,
            'confidence': confidence,
            'buy_probability': buy_proba[1],
            'predicted_rr': max(rr_prediction, 0.01),
            'method': 'ML Model'
        }
        
    except Exception as e:
        st.error(f"❌ ML model error: {e}")
        st.code(traceback.format_exc())
        return simple_signal_generation(df)

def simple_signal_generation(df):
    """Simple signal generation exactly like app.py"""
    try:
        latest = df.iloc[-1]
        
        rsi = latest.get('RSI', 50)
        momentum = latest.get('Momentum', 0)
        price = latest.get('close', 0)
        sma_10 = latest.get('SMA_10', price)
        
        # Simple scoring
        buy_score = 0
        sell_score = 0
        
        if rsi < 30:
            buy_score += 3
        elif rsi > 70:
            sell_score += 3
        elif rsi < 45:
            buy_score += 1
        
        if momentum > 0:
            buy_score += 2
        else:
            sell_score += 1
        
        if price > sma_10:
            buy_score += 1
        else:
            sell_score += 1
        
        if buy_score >= 4:
            action = "BUY"
            confidence = min(buy_score * 15, 85)
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
            'method': 'Simple Rules'
        }
        
    except Exception as e:
        return {
            'action': 'HOLD',
            'confidence': 0,
            'buy_probability': 0.5,
            'predicted_rr': 1.0,
            'method': 'Error Fallback'
        }

def combine_ml_and_groq_signals(ml_signal, groq_signal):
    """Combine signals exactly like app.py"""
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
        'original_confidence': base_confidence
    }

def fetch_latest_candle(groww, symbol, interval_minutes=10, max_candles=50):
    """Fetch candle data exactly like app.py"""
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
        
        # Add technical indicators
        df = compute_technical_indicators(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching  {str(e)}")
        return None

def perform_complete_analysis(groww, selected_symbol, interval_minutes, groq_key, selected_groq_model, account_balance, risk_per_trade, groq_available):
    """Perform complete analysis exactly like app.py"""
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

# Dashboard visualization functions
def create_price_chart(df, signal_data):
    """Create price chart with signals"""
    try:
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        
        # Add moving averages
        if 'SMA_10' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['SMA_10'],
                mode='lines',
                name='SMA 10',
                line=dict(color='orange', width=1)
            ))
        
        if 'EMA_10' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['EMA_10'],
                mode='lines',
                name='EMA 10',
                line=dict(color='purple', width=1)
            ))
        
        # Add signal markers
        final_signal = signal_data['final_signal']
        if final_signal['action'] in ["BUY", "SELL"]:
            color = 'green' if final_signal['action'] == 'BUY' else 'red'
            symbol_marker = 'triangle-up' if final_signal['action'] == 'BUY' else 'triangle-down'
            
            fig.add_trace(go.Scatter(
                x=[df['timestamp'].iloc[-1]],
                y=[signal_data['current_price']],
                mode='markers',
                name=f'{final_signal["action"]} Signal',
                marker=dict(
                    symbol=symbol_marker,
                    size=15,
                    color=color
                )
            ))
            
            # Add risk levels
            risk_levels = signal_data['risk_levels']
            fig.add_hline(y=risk_levels['stop_loss'], line_dash="dash", line_color="red", 
                         annotation_text=f"Stop Loss: ₹{risk_levels['stop_loss']}")
            fig.add_hline(y=risk_levels['take_profit_1'], line_dash="dash", line_color="green", 
                         annotation_text=f"TP1: ₹{risk_levels['take_profit_1']}")
            fig.add_hline(y=risk_levels['take_profit_2'], line_dash="dash", line_color="darkgreen", 
                         annotation_text=f"TP2: ₹{risk_levels['take_profit_2']}")
        
        fig.update_layout(
            title=f'{signal_data["symbol"]} - Price Chart with Signal',
            xaxis_title='Time',
            yaxis_title='Price (₹)',
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating price chart: {e}")
        return None

def display_analysis_results_dashboard(analysis_data):
    """Display analysis results exactly like app.py but with dashboard layout"""
    
    df = analysis_data['df']
    ml_signal = analysis_data['ml_signal']
    groq_signal = analysis_data['groq_signal']
    final_signal = analysis_data['final_signal']
    current_price = analysis_data['current_price']
    risk_levels = analysis_data['risk_levels']
    quantity = analysis_data['quantity']
    
    # Display Results in Dashboard format
    st.markdown("## 📈 Trading Signal Analysis")
    
    # Signal overview (same as app.py)
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
    
    # Trade Analysis (same as app.py but FIXED)
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
            expected_profit = (max_profit * 0.6) - (max_loss * 0.4)
            st.metric("📊 Expected Profit", f"₹{expected_profit:,.0f}")
        
        # Charts section
        chart_col1, chart_col2 = st.columns([2, 1])
        
        with chart_col1:
            price_chart = create_price_chart(df, analysis_data)
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
        
        with chart_col2:
            st.markdown("### 📋 Trade Plan")
            st.write(f"**Action:** {final_signal['action']}")
            st.write(f"**Entry:** ₹{current_price:.2f}")
            st.write(f"**Quantity:** {quantity:,} shares")
            st.write(f"**Investment:** ₹{investment_amount:,.0f}")
            st.write(f"**Stop Loss:** ₹{risk_levels['stop_loss']}")
            st.write(f"**Take Profit 1:** ₹{risk_levels['take_profit_1']}")
            st.write(f"**Take Profit 2:** ₹{risk_levels['take_profit_2']}")
            st.write(f"**Expected Profit:** ₹{expected_profit:,.0f}")
    
    # Technical Analysis (same as app.py)
    st.markdown("### 🔧 Technical Analysis")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    latest = df.iloc[-1]
    
    with tech_col1:
        st.write(f"**RSI:** {latest['RSI']:.1f}")
        st.write(f"**MACD:** {latest['MACD']:.4f}")
        st.write(f"**Signal:** {latest['MACD_Signal']:.4f}")
    
    with tech_col2:
        st.write(f"**SMA 10:** ₹{latest['SMA_10']:.2f}")
        st.write(f"**EMA 10:** ₹{latest['EMA_10']:.2f}")
        st.write(f"**Momentum:** {latest['Momentum']:.2f}")
    
    with tech_col3:
        st.write(f"**Volatility:** {latest['Volatility']:.2f}")
        st.write(f"**Volume Ratio:** {latest['Volume_Ratio']:.2f}x")
        st.write(f"**ATR:** {risk_levels.get('atr', 0):.2f}")
    
    # Recent market data
    st.markdown("### 📋 Recent Market Data")
    display_df = df.tail(10)[['timestamp', 'close', 'open', 'high', 'low', 'volume', 'RSI']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    display_df = display_df.round(2)
    st.dataframe(display_df, use_container_width=True)

# Main Dashboard Function
def run_dashboard():
    """Main dashboard function - identical signal logic as app.py"""
    st.set_page_config(page_title="Trading Signal Dashboard", layout="wide")
    
    st.title("📊 Trading Signal Dashboard")
    st.markdown("### Manual Trading Assistant - Signals & Analysis")
    
    # Sidebar (same as app.py)
    st.sidebar.title("🔐 API Authentication")
    grow_key = st.sidebar.text_input("Groww API token", type="password", key="grow_api_key")
    groq_key = st.sidebar.text_input("Groq API key (optional)", type="password", key="groq_api_key")
    
    # System initialization (same as app.py)
    if not grow_key:
        st.warning("Please enter your Groww API token in the sidebar.")
        st.stop()
    
    # Load models (same as app.py)
    if not st.session_state.models_loaded:
        with st.spinner("Loading ML models..."):
            success, message = load_models_safely()
            if success:
                st.success("✅ " + message)
            else:
                st.warning("⚠️ " + message)
    
    # Initialize Groww API (same as app.py)
    with st.spinner("Initializing Groww API..."):
        groww, instruments_df, init_message = initialize_groww_safely()
    
    if groww is None:
        st.error(f"❌ {init_message}")
        st.stop()
    
    st.success(f"✅ {init_message}")
    
    # Check Groq availability (same as app.py)
    groq_available = False
    groq_models = []
    if groq_key:
        groq_models, groq_error = get_groq_models(groq_key)
        if groq_models:
            groq_available = True
            st.success("✅ Groq LLM available for enhanced signals")
        else:
            st.warning(f"⚠️ Groq unavailable: {groq_error}")
    
    # Symbol selection and settings (same as app.py)
    symbols_list = instruments_df["groww_symbol"].sort_values().unique().tolist()
    default_symbol = "NSE-NIFTY" if "NSE-NIFTY" in symbols_list else symbols_list[0]
    selected_symbol = st.sidebar.selectbox("Select Symbol", symbols_list, index=symbols_list.index(default_symbol))
    
    # Settings (same as app.py)
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Account Settings")
    account_balance = st.sidebar.number_input("Account Balance (₹)", value=100000, min_value=10000, step=10000)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)  # FIXED: Increased default
    
    st.sidebar.subheader("📊 Analysis Settings")
    interval_minutes = st.sidebar.selectbox("Candle Interval", [5, 10, 15, 30, 60], index=1)
    if groq_available:
        selected_groq_model = st.sidebar.selectbox("Groq Model", groq_models, index=0)
    
    # Main Analysis (same as app.py)
    st.markdown("## 🎯 Trading Analysis")
    
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        analyze_clicked = st.button("🔍 Analyze Symbol", type="primary", use_container_width=True)
    
    with button_col2:
        refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True)
    
    # Perform analysis (same logic as app.py)
    if analyze_clicked or refresh_clicked:
        with st.spinner(f"{'Refreshing' if refresh_clicked else 'Analyzing'} {selected_symbol}..."):
            
            # Use the exact same analysis function as app.py
            analysis_data, message = perform_complete_analysis(
                groww, selected_symbol, interval_minutes, 
                groq_key, selected_groq_model if groq_available else None,
                account_balance, risk_per_trade, groq_available
            )
            
            if analysis_data:
                st.session_state.analysis_data = analysis_data
                st.session_state.last_refresh = datetime.now()
                st.success(f"✅ {message}")
                
                # Display results using the same logic as app.py
                display_analysis_results_dashboard(analysis_data)
                
            else:
                st.error(f"❌ {message}")

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

if __name__ == "__main__":
    run_dashboard()
