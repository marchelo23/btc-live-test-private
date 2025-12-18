"""
Data Ingestion Module
Fetches OHLCV data from Binance using CCXT with robust error handling
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import yaml
from loguru import logger
from typing import Optional, Tuple


class DataIngestion:
    """
    Handles data fetching from cryptocurrency exchanges via CCXT
    Optimized for local storage on Parrot OS
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data ingestion

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.exchange = self._initialize_exchange()

        # Setup directories
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange connection"""
        try:
            exchange_name = self.config['data']['exchange']
            exchange_class = getattr(ccxt, exchange_name)

            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })

            logger.info(f"Connected to {exchange_name}")
            return exchange

        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

    def fetch_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        lookback_days: Optional[int] = None,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with automatic pagination and retry logic

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '30m', '1h')
            lookback_days: Number of days to fetch
            max_retries: Maximum retry attempts on failure

        Returns:
            DataFrame with OHLCV data
        """
        # Use config defaults if not specified
        symbol = symbol or self.config['data']['symbol']
        timeframe = timeframe or self.config['data']['base_timeframe']
        lookback_days = lookback_days or self.config['data']['lookback_days']

        logger.info(f"Fetching {lookback_days} days of {timeframe} data for {symbol}")

        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            since = int(start_time.timestamp() * 1000)  # Milliseconds

            all_candles = []
            current_since = since

            # Fetch data in chunks
            while current_since < int(end_time.timestamp() * 1000):
                for attempt in range(max_retries):
                    try:
                        candles = self.exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=current_since,
                            limit=1000  # Max for most exchanges
                        )

                        if not candles:
                            break

                        all_candles.extend(candles)
                        current_since = candles[-1][0] + 1

                        logger.debug(f"Fetched {len(candles)} candles (Total: {len(all_candles)})")

                        # Rate limiting
                        time.sleep(self.exchange.rateLimit / 1000)
                        break

                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                            time.sleep(wait_time)
                        else:
                            raise

            if not all_candles:
                logger.error("No data fetched")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

            # Validate data quality
            is_valid, message = self._validate_data(df, timeframe)
            logger.info(f"Data validation: {message}")

            logger.info(f"✓ Fetched {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}")
            return None

    def _validate_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[bool, str]:
        """
        Validate data quality

        Args:
            df: DataFrame to validate
            timeframe: Expected timeframe

        Returns:
            Tuple of (is_valid, message)
        """
        issues = []

        # Check for missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        if missing_pct.any():
            issues.append(f"Missing values: {missing_pct[missing_pct > 0].to_dict()}")

        # Check for zero volumes
        zero_volume_pct = (df['volume'] == 0).sum() / len(df) * 100
        if zero_volume_pct > 5:
            issues.append(f"High zero-volume candles: {zero_volume_pct:.2f}%")

        # Check for price anomalies
        price_changes = df['close'].pct_change().abs()
        extreme_moves = (price_changes > 0.1).sum()  # >10% move
        if extreme_moves > len(df) * 0.01:
            issues.append(f"Unusual price spikes: {extreme_moves} instances")

        # Check time gaps
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()

        # Map timeframe to expected timedelta
        timeframe_map = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '30m': pd.Timedelta(minutes=30),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1)
        }

        expected_diff = timeframe_map.get(timeframe, pd.Timedelta(minutes=30))
        gaps = (time_diffs > expected_diff * 2).sum()
        if gaps > 0:
            issues.append(f"Time gaps detected: {gaps} instances")

        if issues:
            return False, "; ".join(issues)
        else:
            return True, "Data quality: OK"

    def save_raw_data(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Save raw data to CSV

        Args:
            df: DataFrame to save
            filename: Custom filename (optional)

        Returns:
            Path where data was saved
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                symbol = self.config['data']['symbol'].replace('/', '_')
                filename = f"{symbol}_{timestamp}.csv"

            filepath = self.raw_dir / filename
            df.to_csv(filepath, index=False)

            logger.info(f"✓ Raw data saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save raw data: {e}")
            raise

    def load_latest_raw_data(self) -> Optional[pd.DataFrame]:
        """
        Load the most recent raw data file

        Returns:
            DataFrame with raw data or None if not found
        """
        try:
            # Find latest CSV file
            csv_files = sorted(self.raw_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)

            if not csv_files:
                logger.warning("No raw data files found")
                return None

            latest_file = csv_files[0]
            logger.info(f"Loading raw data from {latest_file}")

            df = pd.read_csv(latest_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            logger.info(f"✓ Loaded {len(df)} rows from {latest_file.name}")
            return df

        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            return None

    def fetch_and_save(self) -> Optional[str]:
        """
        Convenience method: Fetch data and save it

        Returns:
            Path to saved file
        """
        df = self.fetch_ohlcv()

        if df is not None:
            return self.save_raw_data(df)
        else:
            logger.error("Data fetching failed")
            return None


def main():
    """CLI entry point for data ingestion"""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch BTC OHLCV data")
    parser.add_argument('--symbol', type=str, default=None, help='Trading pair (e.g., BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default=None, help='Timeframe (e.g., 30m, 1h)')
    parser.add_argument('--days', type=int, default=None, help='Days to fetch')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')

    args = parser.parse_args()

    # Setup logging
    logger.add("logs/ingestion.log", rotation="1 day", retention="7 days")

    # Fetch data
    ingestion = DataIngestion(config_path=args.config)
    filepath = ingestion.fetch_and_save()

    if filepath:
        print(f"✓ Data saved to: {filepath}")
        return 0
    else:
        print("✗ Data fetching failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
