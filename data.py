import os
import pandas as pd
import datetime
import pytz
from dotenv import load_dotenv
from growwapi import GrowwAPI
import warnings
import time

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
AUTH_TOKEN = os.getenv("GROWW_AUTH_TOKEN")

def prepare_df(raw_data):
    """Convert raw candle data to pandas DataFrame"""
    df = pd.DataFrame(raw_data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    
    # Convert price columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def initialize_groww_api(auth_token, instruments_path="instruments.csv"):
    """Initialize Groww API with instruments data"""
    if not auth_token:
        raise ValueError("AUTH_TOKEN is required. Please set it in your .env file")
    
    groww = GrowwAPI(auth_token)
    
    try:
        instruments_df = pd.read_csv(instruments_path, low_memory=False)
        groww.instruments = instruments_df
        groww._load_instruments = lambda: None
        groww._download_and_load_instruments = lambda: instruments_df
        print(f"✅ Loaded {len(instruments_df)} instruments from {instruments_path}")
    except FileNotFoundError:
        print(f"❌ Instruments file not found: {instruments_path}")
        raise
    
    return groww

def fetch_candles_chunk(groww, symbol, start, end, interval):
    """Fetch candle data for a specific time period"""
    # Rate limits for different intervals (in days)
    max_daily = {
        1: 7,       # 1 min - 7 days
        5: 15,      # 5 min - 15 days
        10: 30,     # 10 min - 30 days
        15: 31,     # 15 min - 31 days
        30: 90,     # 30 min - 90 days
        60: 150,    # 60 min - 150 days
        240: 365,   # 4h - 365 days
        1440: 1080, # 1d - 1080 days (~3y)
    }
    
    days = (end - start).days
    if interval not in max_daily or days > max_daily[interval]:
        print(f"⚠️ Skipping: {interval}m interval not allowed for {days} days (max: {max_daily.get(interval, 'N/A')})")
        return None
    
    try:
        print(f"📡 Fetching {symbol}: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')} ({interval}m)")
        
        data = groww.get_historical_candle_data(
            trading_symbol=symbol,
            exchange=groww.EXCHANGE_NSE,
            segment=groww.SEGMENT_CASH,
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
            interval_in_minutes=interval
        )
        
        time.sleep(2)  # Rate limiting to avoid API restrictions
        
        if data and data.get("candles"):
            print(f"✅ Retrieved {len(data['candles'])} candles")
            return data
        else:
            print(f"⚠️ No data returned for {symbol}")
            return None
            
    except Exception as e:
        print(f"❌ API error for {symbol}: {e}")
        return None

def load_data(symbol="NIFTY", days_back=360):
    """Load historical data with multiple timeframes"""
    print(f"🚀 Loading data for {symbol}")
    
    if not AUTH_TOKEN:
        raise ValueError("GROWW_AUTH_TOKEN not found in environment variables")
    
    # Initialize API
    groww = initialize_groww_api(AUTH_TOKEN)
    
    # Set timezone and current time
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(tz=ist).replace(hour=15, minute=15, second=0, microsecond=0)
    
    # Define data collection periods (label, days_ago_start, days_ago_end, interval_minutes)
    periods = [
        ("df_live", 0, 30, 10),      # Recent data with 10-min candles
        ("df_1", 30, 60, 15),        # Month ago with 15-min candles
        ("df_2", 60, 90, 30),        # 2 months ago with 30-min candles
        ("df_3", 90, 180, 1440),     # 3-6 months ago with daily candles
        ("df_4", 180, min(360, days_back), 1440),  # Older data with daily candles
    ]
    
    dfs = {}
    
    for label, ago_start, ago_end, interval in periods:
        start_date = now - datetime.timedelta(days=ago_end)
        end_date = now - datetime.timedelta(days=ago_start)
        
        print(f"\n📊 Processing {label}...")
        chunk = fetch_candles_chunk(groww, symbol, start_date, end_date, interval)
        
        if chunk is not None:
            df = prepare_df(chunk)
            dfs[label] = df
            print(f"✅ {label}: {len(df)} rows | Timeframe: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print(f"❌ {label}: Failed to fetch data")
    
    if not dfs:
        print("❌ No data collected. Check your API token and symbol.")
        return groww, {}
    
    print(f"\n🎉 Successfully collected {len(dfs)} data segments")
    return groww, dfs

def create_master_dataframe(dfs):
    """Combine all dataframes into a single master dataframe"""
    if not dfs:
        print("❌ No dataframes to combine")
        return pd.DataFrame()
    
    print("🔄 Creating master dataframe...")
    
    # Combine all dataframes
    all_dfs = list(dfs.values())
    df_master = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by timestamp and remove duplicates
    df_master = df_master.sort_values(by="timestamp")
    df_master = df_master.drop_duplicates(subset=['timestamp'], keep='last')
    df_master = df_master.reset_index(drop=True)
    
    print(f"✅ Master dataframe created: {len(df_master)} rows")
    print(f"📅 Date range: {df_master['timestamp'].min()} to {df_master['timestamp'].max()}")
    
    return df_master

def save_data_to_csv(dfs, symbol="NIFTY"):
    """Save collected data to CSV files"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual segments
    for label, df in dfs.items():
        filename = f"data_{symbol}_{label}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Saved {filename}")
    
    # Save master dataframe
    df_master = create_master_dataframe(dfs)
    if not df_master.empty:
        master_filename = f"data_{symbol}_master_{timestamp}.csv"
        df_master.to_csv(master_filename, index=False)
        print(f"💾 Saved {master_filename}")
        
        return master_filename
    
    return None

if __name__ == "__main__":
    print("🚀 Data Collection Script")
    print("=" * 50)
    
    try:
        # Load data for NIFTY
        groww_api, data_frames = load_data("NIFTY")
        
        if data_frames:
            # Display summary
            print("\n📋 Data Summary:")
            for label, df in data_frames.items():
                print(f"  {label}: {len(df)} rows | Columns: {list(df.columns)}")
                if len(df) > 0:
                    print(f"    Latest price: ₹{df['close'].iloc[-1]:.2f}")
                    print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print()
            
            # Create and display master dataframe info
            df_master = create_master_dataframe(data_frames)
            if not df_master.empty:
                print(f"📊 Master DataFrame Summary:")
                print(f"   Total rows: {len(df_master)}")
                print(f"   Columns: {list(df_master.columns)}")
                print(f"   Latest close: ₹{df_master['close'].iloc[-1]:.2f}")
                print(f"   Price range: ₹{df_master['close'].min():.2f} - ₹{df_master['close'].max():.2f}")
            
            # Save to CSV
            save_option = input("\n💾 Save data to CSV? (y/n): ").lower().strip()
            if save_option == 'y':
                saved_file = save_data_to_csv(data_frames)
                if saved_file:
                    print(f"✅ Data saved successfully!")
        else:
            print("❌ No data collected. Please check your setup.")
            
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        import traceback
        traceback.print_exc()
