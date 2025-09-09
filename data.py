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
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(1)
    return df

def initialize_groww_api(auth_token, instruments_path="instruments.csv"):
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
    max_daily = {
        1: 7, 5: 15, 10: 30, 15: 31, 30: 90, 60: 150, 240: 365, 1440: 1080,
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
        time.sleep(2)
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
    print(f"🚀 Loading data for {symbol}")
    if not AUTH_TOKEN:
        raise ValueError("GROWW_AUTH_TOKEN not found in environment variables")
    groww = initialize_groww_api(AUTH_TOKEN)
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(tz=ist).replace(hour=15, minute=15, second=0, microsecond=0)
    periods = [
        ("df_live", 0, 15, 5),
        ("df_1", 15, 45, 10),
        ("df_2", 60, 90, 30),
        ("df_3", 90, 180, 1440),
        ("df_4", 180, min(360, days_back), 1440),
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
    if not dfs:
        print("❌ No dataframes to combine")
        return pd.DataFrame()
    print("🔄 Creating master dataframe...")
    all_dfs = list(dfs.values())
    df_master = pd.concat(all_dfs, ignore_index=True)
    df_master = df_master.sort_values(by="timestamp")
    df_master = df_master.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
    print(f"✅ Master dataframe created: {len(df_master)} rows")
    print(f"📅 Date range: {df_master['timestamp'].min()} to {df_master['timestamp'].max()}")
    return df_master

def add_signal_column(df):
    # Updated to 'Buy_Signal' with matching naming for model trainer
    if 'Buy_Score' not in df.columns:
        # Fallback: use price direction (just for demonstration)
        df['Buy_Signal'] = (df['close'] > df['open']).astype(int)
    else:
        df['Buy_Signal'] = (df['Buy_Score'] > 0.7).astype(int)
    return df

def create_balanced_dataset(df, label_col="Buy_Signal"):
    buy_df = df[df[label_col] == 1]
    sell_df = df[df[label_col] == 0]
    min_samples = min(len(buy_df), len(sell_df))
    if min_samples == 0:
        print("⚠️ Cannot balance dataset, one class is empty.")
        return df  # Return unbalanced in this case
    buy_sample = buy_df.sample(n=min_samples, random_state=42)
    sell_sample = sell_df.sample(n=min_samples, random_state=42)
    df_balanced = pd.concat([buy_sample, sell_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✅ Balanced dataset: {min_samples} buys, {min_samples} sells, {len(df_balanced)} total samples")
    return df_balanced

def save_data_to_csv(dfs, symbol="NSE-NIFTY"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for label, df in dfs.items():
        filename = f"data_{symbol}_{label}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Saved {filename}")
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
        groww_api, data_frames = load_data("NIFTY")
        if data_frames:
            print("\n📋 Data Summary:")
            for label, df in data_frames.items():
                print(f"  {label}: {len(df)} rows | Columns: {list(df.columns)}")
                if len(df) > 0:
                    print(f"    Latest price: ₹{df['close'].iloc[-1]:.2f}")
                    print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print()
            df_master = create_master_dataframe(data_frames)
            df_master = add_signal_column(df_master)
            print(df_master['Buy_Signal'].value_counts())
            df_balanced = create_balanced_dataset(df_master, label_col="Buy_Signal")
            print(df_balanced['Buy_Signal'].value_counts())
            save_option = input("\n💾 Save data to CSV, including balanced data? (y/n): ").strip().lower()
            if save_option == 'y':
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                df_master.to_csv(f"data_NIFTY_master_{timestamp}.csv", index=False)
                df_balanced.to_csv(f"data_NIFTY_balanced_{timestamp}.csv", index=False)
                print(f"✅ Data and balanced data saved successfully.")
        else:
            print("❌ No data collected. Please check your setup.")
    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        import traceback
        traceback.print_exc()
