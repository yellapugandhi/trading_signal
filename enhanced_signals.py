import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from datetime import datetime

@dataclass
class TradingSignal:
    symbol: str
    timestamp: datetime
    action: str  # BUY, SELL, HOLD
    confidence: float
    current_price: float
    quantity: int
    investment_amount: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    max_profit: float
    max_loss: float
    expected_profit: float
    technical_factors: Dict
    model_confidence: float
    llm_signal: str

class EnhancedTradingEngine:
    def __init__(self, buy_model, rr_model, account_balance=100000):
        self.buy_model = buy_model
        self.rr_model = rr_model
        self.account_balance = account_balance
        self.risk_per_trade = 0.015  # 1.5% risk per trade
        self.max_position_size = 0.12  # Max 12% per position
        
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        if stop_loss == 0 or entry_price == stop_loss:
            return 0
            
        risk_amount = self.account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        position_size = int(risk_amount / price_risk)
        max_shares_by_value = int((self.account_balance * self.max_position_size) / entry_price)
        
        final_size = min(position_size, max_shares_by_value, 2000)  # Cap to 2000 shares
        return max(final_size, 0)
    
    def calculate_risk_levels(self, df: pd.DataFrame, action: str, current_price: float) -> Dict:
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(14).mean().iloc[-1]
        
        features = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility",
                   "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal"]
        latest_features = df[features].iloc[-1:].fillna(0)
        predicted_rr = self.rr_model.predict(latest_features)[0]
        predicted_rr = max(predicted_rr, 0.02)
        
        if action == "BUY":
            stop_loss = round(current_price - (1.5 * atr), 2)
            risk_per_share = current_price - stop_loss
            tp1 = round(current_price + (risk_per_share * 1.5), 2)
            tp2 = round(current_price + (risk_per_share * (predicted_rr * 100)), 2)
        else:
            stop_loss = round(current_price + (1.5 * atr), 2)
            risk_per_share = stop_loss - current_price
            tp1 = round(current_price - (risk_per_share * 1.5), 2)
            tp2 = round(current_price - (risk_per_share * (predicted_rr * 100)), 2)
        
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(tp2 - current_price)
        actual_rr = round(reward_amount / risk_amount, 2) if risk_amount > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'risk_reward_ratio': actual_rr,
            'atr': round(atr, 2),
            'predicted_rr': round(predicted_rr, 4)
        }
    
    def calculate_profit_projections(self, entry_price: float, quantity: int, 
                                   stop_loss: float, take_profit_2: float) -> Dict:
        if quantity == 0:
            return {'max_profit': 0, 'max_loss': 0, 'expected_profit': 0, 'break_even_rate': 0}
        
        max_loss = abs(entry_price - stop_loss) * quantity
        max_profit = abs(take_profit_2 - entry_price) * quantity
        win_rate = 0.62
        
        expected_profit = (max_profit * win_rate) - (max_loss * (1 - win_rate))
        break_even_rate = max_loss / (max_profit + max_loss) * 100 if (max_profit + max_loss) > 0 else 0
        
        return {
            'max_profit': round(max_profit, 2),
            'max_loss': round(max_loss, 2),
            'expected_profit': round(expected_profit, 2),
            'break_even_rate': round(break_even_rate, 1),
            'win_rate_assumption': win_rate
        }
    
    def get_technical_factors(self, df: pd.DataFrame) -> Dict:
        latest = df.iloc[-1]
        
        rsi_status = "Oversold" if latest['RSI'] < 30 else "Overbought" if latest['RSI'] > 70 else "Neutral"
        momentum_status = "Bullish" if latest['Momentum'] > 0 else "Bearish"
        macd_status = "Bullish" if latest['MACD'] > latest['MACD_Signal'] else "Bearish"
        trend_status = "Uptrend" if latest['close'] > latest['SMA_10'] > latest['EMA_10'] else "Downtrend"
        
        return {
            'rsi': round(latest['RSI'], 1),
            'rsi_status': rsi_status,
            'momentum': round(latest['Momentum'], 2),
            'momentum_status': momentum_status,
            'macd': round(latest['MACD'], 4),
            'macd_signal': round(latest['MACD_Signal'], 4),
            'macd_status': macd_status,
            'trend_status': trend_status,
            'volatility': round(latest['Volatility'], 2),
            'price_vs_sma': round(((latest['close'] / latest['SMA_10']) - 1) * 100, 2)
        }
    
    def generate_enhanced_signal(self, df: pd.DataFrame, symbol: str, 
                               llm_signal: str = "UNKNOWN") -> TradingSignal:
        features = ["SMA_10", "EMA_10", "RSI", "Momentum", "Volatility",
                   "Lag_Close", "Lag_Momentum", "MACD", "MACD_Signal"]
        
        latest_features = df[features].iloc[-1:].fillna(0)
        buy_proba = self.buy_model.predict_proba(latest_features)[0]
        model_confidence = buy_proba[1] * 100
        
        if model_confidence >= 65:
            action = "BUY"
            confidence = model_confidence
        elif model_confidence <= 35:
            action = "SELL"
            confidence = 100 - model_confidence
        else:
            action = "HOLD"
            confidence = 50
        
        current_price = df['close'].iloc[-1]
        timestamp = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
        
        risk_levels = self.calculate_risk_levels(df, action, current_price)
        quantity = 0
        if action in ["BUY", "SELL"]:
            quantity = self.calculate_position_size(current_price, risk_levels['stop_loss'])
        
        investment_amount = current_price * quantity
        profit_proj = self.calculate_profit_projections(
            current_price, quantity, risk_levels['stop_loss'], risk_levels['take_profit_2']
        )
        
        technical_factors = self.get_technical_factors(df)
        
        return TradingSignal(
            symbol=symbol,
            timestamp=timestamp,
            action=action,
            confidence=round(confidence, 1),
            current_price=round(current_price, 2),
            quantity=quantity,
            investment_amount=round(investment_amount, 2),
            stop_loss=risk_levels['stop_loss'],
            take_profit_1=risk_levels['take_profit_1'],
            take_profit_2=risk_levels['take_profit_2'],
            risk_reward_ratio=risk_levels['risk_reward_ratio'],
            max_profit=profit_proj['max_profit'],
            max_loss=profit_proj['max_loss'],
            expected_profit=profit_proj['expected_profit'],
            technical_factors=technical_factors,
            model_confidence=round(model_confidence, 1),
            llm_signal=llm_signal
        )
