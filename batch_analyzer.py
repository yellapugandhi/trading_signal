import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
from typing import List, Optional
from enhanced_signals import EnhancedTradingEngine, TradingSignal

warnings.filterwarnings("ignore")

class BatchAnalyzer:
    def __init__(self, enhanced_engine: EnhancedTradingEngine, groww_api, instruments_df):
        self.engine = enhanced_engine
        self.groww = groww_api
        self.instruments = instruments_df
        
    def feature_engineering(self, df):
        """Apply feature engineering to dataframe"""
        df['SMA_10'] = df['close'].rolling(10).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['Momentum'] = df['close'] - df['close'].shift(5)
        df['Volatility'] = df['close'].rolling(10).std()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / (loss.replace(0, 1e-8))
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Lag_Close'] = df['close'].shift(1)
        df['Lag_Momentum'] = df['Momentum'].shift(1)
        
        # MACD calculation
        ema1 = df['close'].ewm(span=12, adjust=False).mean()
        ema2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema1 - ema2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def get_market_times(self, now):
        """Get market open and close times"""
        ist = ZoneInfo("Asia/Kolkata")
        today = now.date()
        market_open = datetime.combine(today, datetime.strptime("09:15", "%H:%M").time(), tzinfo=ist)
        market_close = datetime.combine(today, datetime.strptime("15:30", "%H:%M").time(), tzinfo=ist)
        return market_open, market_close

    def fetch_symbol_data(self, symbol: str, interval_minutes: int = 10, max_candles: int = 50) -> Optional[pd.DataFrame]:
        """Fetch historical data for a single symbol"""
        try:
            ist = ZoneInfo("Asia/Kolkata")
            now = datetime.now(ist)
            market_open, market_close = self.get_market_times(now)
            
            if now < market_open:
                prev_day = now - timedelta(days=1)
                _, yest_close = self.get_market_times(prev_day)
                end_time = yest_close
            elif now > market_close:
                end_time = market_close
            else:
                end_time = now.replace(second=0, microsecond=0)
            
            start_time = end_time - timedelta(minutes=interval_minutes * max_candles)
            
            # Get instrument details
            selected = self.instruments[self.instruments['groww_symbol'] == symbol].iloc[0].to_dict()
            
            data = self.groww.get_historical_candle_data(
                trading_symbol=selected['trading_symbol'],
                exchange=selected['exchange'],
                segment=selected['segment'],
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                interval_in_minutes=interval_minutes
            )
            
            if not isinstance(data, dict) or not data.get('candles'):
                return None
            
            df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.ffill().bfill()
            df = self.feature_engineering(df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def analyze_single_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Analyze a single symbol and return trading signal"""
        try:
            df = self.fetch_symbol_data(symbol)
            if df is None or len(df) < 20:
                return None
            
            # Generate signal with simplified LLM (you can enhance this)
            signal = self.engine.generate_enhanced_signal(df, symbol, "UNKNOWN")
            
            # Only return actionable signals
            if signal.action in ["BUY", "SELL"] and signal.confidence > 60:
                return signal
            
            return None
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def batch_analyze(self, symbols: List[str], max_workers: int = 3) -> List[TradingSignal]:
        """Analyze multiple symbols in parallel"""
        print(f"🔍 Starting batch analysis for {len(symbols)} symbols...")
        
        results = []
        processed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.analyze_single_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                processed += 1
                
                try:
                    signal = future.result(timeout=30)
                    if signal:
                        results.append(signal)
                        print(f"✅ {symbol}: {signal.action} ({signal.confidence}%)")
                    else:
                        print(f"⚪ {symbol}: No actionable signal")
                        
                except Exception as e:
                    print(f"❌ {symbol}: Error - {e}")
                
                print(f"Progress: {processed}/{len(symbols)}")
                time.sleep(0.5)  # Rate limiting
        
        return results
    
    def create_signals_report(self, signals: List[TradingSignal]) -> pd.DataFrame:
        """Create a comprehensive report of all signals"""
        if not signals:
            return pd.DataFrame()
        
        report_data = []
        for signal in signals:
            report_data.append({
                'Symbol': signal.symbol,
                'Action': signal.action,
                'Confidence': f"{signal.confidence}%",
                'Current_Price': f"₹{signal.current_price}",
                'Quantity': signal.quantity,
                'Investment': f"₹{signal.investment_amount:,.0f}",
                'Stop_Loss': f"₹{signal.stop_loss}",
                'Take_Profit': f"₹{signal.take_profit_2}",
                'Risk_Reward': f"1:{signal.risk_reward_ratio}",
                'Expected_Profit': f"₹{signal.expected_profit:,.0f}",
                'Max_Profit': f"₹{signal.max_profit:,.0f}",
                'Max_Loss': f"₹{signal.max_loss:,.0f}",
                'RSI': signal.technical_factors['rsi'],
                'Trend': signal.technical_factors['trend_status'],
                'Timestamp': signal.timestamp.strftime('%H:%M:%S')
            })
        
        df = pd.DataFrame(report_data)
        return df.sort_values('Confidence', ascending=False)
    
    def save_signals_report(self, signals: List[TradingSignal], filename: str = None):
        """Save signals report to CSV"""
        if not filename:
            filename = f"batch_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        report_df = self.create_signals_report(signals)
        
        if not report_df.empty:
            report_df.to_csv(filename, index=False)
            print(f"📁 Report saved to {filename}")
            return filename
        else:
            print("📭 No signals to save")
            return None
    
    def get_top_opportunities(self, signals: List[TradingSignal], top_n: int = 5) -> List[TradingSignal]:
        """Get top trading opportunities based on confidence and expected profit"""
        if not signals:
            return []
        
        # Sort by confidence score and expected profit
        sorted_signals = sorted(
            signals,
            key=lambda x: (x.confidence, x.expected_profit),
            reverse=True
        )
        
        return sorted_signals[:top_n]
    
    def print_summary(self, signals: List[TradingSignal]):
        """Print a summary of the batch analysis"""
        if not signals:
            print("📭 No actionable signals found")
            return
        
        buy_signals = [s for s in signals if s.action == "BUY"]
        sell_signals = [s for s in signals if s.action == "SELL"]
        
        total_investment = sum(s.investment_amount for s in signals)
        total_expected_profit = sum(s.expected_profit for s in signals)
        
        print("\n" + "="*60)
        print("📊 BATCH ANALYSIS SUMMARY")
        print("="*60)
        print(f"🎯 Total Signals Found: {len(signals)}")
        print(f"🟢 Buy Signals: {len(buy_signals)}")
        print(f"🔴 Sell Signals: {len(sell_signals)}")
        print(f"💰 Total Investment Required: ₹{total_investment:,.0f}")
        print(f"📈 Total Expected Profit: ₹{total_expected_profit:,.0f}")
        
        if signals:
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            print(f"📊 Average Confidence: {avg_confidence:.1f}%")
        
        print("\n🏆 TOP 3 OPPORTUNITIES:")
        top_3 = self.get_top_opportunities(signals, 3)
        for i, signal in enumerate(top_3, 1):
            print(f"{i}. {signal.symbol}: {signal.action} ({signal.confidence}%) - Expected: ₹{signal.expected_profit:,.0f}")
        
        print("="*60)

def get_popular_symbols():
    """Return a list of popular trading symbols"""
    return [
        "NSE-RELIANCE", "NSE-TCS", "NSE-INFY", "NSE-HDFCBANK", "NSE-ICICIBANK",
        "NSE-HINDUNILVR", "NSE-ITC", "NSE-KOTAKBANK", "NSE-LT", "NSE-ASIANPAINT",
        "NSE-MARUTI", "NSE-AXISBANK", "NSE-SUNPHARMA", "NSE-TITAN", "NSE-ULTRACEMCO",
        "NSE-BAJFINANCE", "NSE-NESTLEIND", "NSE-POWERGRID", "NSE-NTPC", "NSE-ONGC"
    ]

if __name__ == "__main__":
    print("🚀 Batch Analyzer Test")
    print("Note: This requires your Groww API setup to run")
    
    # Example usage:
    # from growwapi import GrowwAPI
    # import pandas as pd
    # from strategy_1_model import buy_model, rr_model
    # 
    # groww = GrowwAPI("your_token")
    # instruments_df = pd.read_csv("instruments.csv")
    # engine = EnhancedTradingEngine(buy_model, rr_model, 100000)
    # 
    # analyzer = BatchAnalyzer(engine, groww, instruments_df)
    # symbols = get_popular_symbols()[:10]  # Test with 10 symbols
    # signals = analyzer.batch_analyze(symbols)
    # analyzer.print_summary(signals)
    # analyzer.save_signals_report(signals)
