"""
Live Prediction Script for GitHub Actions
Runs every 30 minutes, replicating historical CSV format
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import DataIngestion
from src.inference import ModelInference

# Configuration
LOG_FILE = 'bitacora_new_models.csv'
HORIZONS = {
    '30min': '30m',
    '60min': '1h',
    '180min': '3h',
    '360min': '6h',
    '720min': '12h'
}


def load_or_create_bitacora():
    """Load existing bitacora or create new one"""
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df['timestamp_pred'] = pd.to_datetime(df['timestamp_pred'])
        df['target_time'] = pd.to_datetime(df['target_time'])
        print(f"✅ Loaded {len(df)} existing predictions")
        return df
    else:
        # Create new bitacora with historical CSV format
        df = pd.DataFrame(columns=[
            'timestamp_pred',
            'timeframe',
            'entry_price',
            'predicted_price',
            'direction_pred',
            'target_time',
            'actual_price',
            'error_abs',
            'status'
        ])
        print("✅ Created new bitacora")
        return df


def validate_pending_predictions(df):
    """Validate pending predictions that reached target time"""
    if df.empty:
        return df

    pending = df[df['status'] == 'PENDING'].copy()
    if pending.empty:
        print("ℹ️  No pending predictions to validate")
        return df

    print(f"\n🔍 Validating {len(pending)} pending predictions...")

    # Fetch current data for validation
    try:
        # Use direct ccxt call with limit to avoid rate limiting
        import ccxt
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv('BTC/USD', '30m', limit=50)  # 1 day = 48 candles

        df_current = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df_current['timestamp'] = pd.to_datetime(df_current['timestamp'], unit='ms')
    except Exception as e:
        print(f"⚠️  Could not fetch data for validation: {e}")
        return df

    now = datetime.now()
    validated_count = 0

    for idx, row in pending.iterrows():
        target_time = pd.to_datetime(row['target_time'])

        # Only validate if target time has passed
        if now >= target_time:
            # Find closest price to target time
            df_current['timestamp'] = pd.to_datetime(df_current['timestamp'])
            time_diff = abs(df_current['timestamp'] - target_time)
            closest_idx = time_diff.idxmin()
            actual_price = df_current.loc[closest_idx, 'close']

            # Calculate error
            error_abs = abs(actual_price - row['predicted_price'])

            # Update row
            df.loc[idx, 'actual_price'] = actual_price
            df.loc[idx, 'error_abs'] = error_abs
            df.loc[idx, 'status'] = 'COMPLETED'

            validated_count += 1

    if validated_count > 0:
        print(f"✅ Validated {validated_count} predictions")

    return df


def run_live_prediction():
    """Main prediction function"""
    print("="*80)
    print("🚀 LIVE PREDICTION - NEW MODELS")
    print(f"⏰ Timestamp: {datetime.now()}")
    print("="*80)

    # Load/create bitacora
    log_df = load_or_create_bitacora()

    # Validate pending predictions first
    log_df = validate_pending_predictions(log_df)

    # Fetch fresh data
    print("\n📥 Fetching latest BTC data...")
    ingestion = DataIngestion()
    try:
        # Use simple limit approach (avoids rate limiting issues)
        # 200 candles of 30min = ~4 days of data
        import ccxt
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv('BTC/USD', '30m', limit=200)

        df_raw = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')

        if df_raw is None or df_raw.empty:
            print(f"❌ Error: No data fetched from exchange")
            return

        current_price = df_raw.iloc[-1]['close']
        print(f"✅ Current BTC price: ${current_price:,.2f}")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load models and make predictions
    print("\n🤖 Loading models and making predictions...")
    inference = ModelInference()

    # Prepare features from raw data
    features = inference.prepare_features_from_raw(df_raw)
    if features is None:
        print("❌ Error: Failed to prepare features")
        return

    timestamp_pred = datetime.now()
    new_predictions = []

    for horizon_key, timeframe_label in HORIZONS.items():
        try:
            # Extract integer horizon from string (e.g., '30min' -> 30)
            horizon_int = int(horizon_key.replace('min', ''))

            # Make prediction with prepared features
            result = inference.predict_single_horizon(features, horizon_int)

            if result is None:
                print(f"⚠️  Skipping {timeframe_label}: No prediction")
                continue

            predicted_price = result['predicted_price']
            direction = 'UP' if predicted_price > current_price else 'DOWN'

            # Calculate target time
            target_time = timestamp_pred + timedelta(minutes=horizon_int)

            # Create prediction entry matching historical format
            prediction = {
                'timestamp_pred': timestamp_pred,
                'timeframe': timeframe_label,
                'entry_price': current_price,
                'predicted_price': predicted_price,
                'direction_pred': direction,
                'target_time': target_time,
                'actual_price': np.nan,
                'error_abs': np.nan,
                'status': 'PENDING'
            }

            new_predictions.append(prediction)

            change_pct = ((predicted_price - current_price) / current_price) * 100
            print(f"  ✅ {timeframe_label:4s}: ${predicted_price:,.2f} ({direction}) [{change_pct:+.2f}%]")

        except Exception as e:
            print(f"  ❌ {timeframe_label}: Error - {e}")
            continue

    # Append new predictions
    if new_predictions:
        new_df = pd.DataFrame(new_predictions)
        log_df = pd.concat([log_df, new_df], ignore_index=True)
        print(f"\n✅ Added {len(new_predictions)} new predictions")

    # Save bitacora
    log_df.to_csv(LOG_FILE, index=False)
    print(f"💾 Saved to: {LOG_FILE}")

    # Show statistics
    print("\n" + "="*80)
    print("📊 STATISTICS")
    print("="*80)

    total_predictions = len(log_df)
    pending = len(log_df[log_df['status'] == 'PENDING'])
    completed = len(log_df[log_df['status'] == 'COMPLETED'])

    print(f"Total predictions: {total_predictions}")
    print(f"  - Pending: {pending}")
    print(f"  - Completed: {completed}")

    if completed > 0:
        completed_df = log_df[log_df['status'] == 'COMPLETED']
        avg_error = completed_df['error_abs'].mean()
        print(f"\nAverage Error (completed): ${avg_error:,.2f}")

        # Win rate by timeframe
        print("\nBy Timeframe:")
        for timeframe in ['30m', '1h', '3h', '6h', '12h']:
            tf_data = completed_df[completed_df['timeframe'] == timeframe]
            if len(tf_data) > 0:
                avg_err = tf_data['error_abs'].mean()
                print(f"  {timeframe:4s}: {len(tf_data):3d} ops | Avg Error: ${avg_err:,.2f}")

    print("\n✅ Prediction cycle completed!")
    print("="*80)


if __name__ == "__main__":
    run_live_prediction()
