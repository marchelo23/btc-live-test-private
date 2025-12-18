"""
Feature Engineering Module
Creates technical indicators and multi-horizon targets using pandas-ta
Optimized for vectorized operations (fast on Parrot OS)
"""

import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
from pathlib import Path
import yaml
from loguru import logger
from typing import List, Optional, Tuple


class FeatureEngineer:
    """
    Feature engineering for cryptocurrency price prediction
    Uses ta library for technical indicators (pure Python, easy install)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize feature engineer

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added features
        """
        logger.info("Creating features...")

        df = df.copy()

        # 1. Price-based features
        df = self._add_price_features(df)

        # 2. Technical indicators
        df = self._add_technical_indicators(df)

        # 3. Volume features
        df = self._add_volume_features(df)

        # 4. Volatility features
        df = self._add_volatility_features(df)

        # 5. Time-based features
        df = self._add_time_features(df)

        # 6. Lag features
        df = self._add_lag_features(df)

        # Remove NaN rows
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values")

        logger.info(f"✓ Created {len(df.columns)} features")

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features"""
        # Returns (log returns for stability)
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        # Intrabar price movements
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']

        # Price position within candle
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using ta library"""
        config = self.config['features']['indicators']

        # RSI (Relative Strength Index)
        rsi_indicator = momentum.RSIIndicator(close=df['close'], window=config['rsi_period'])
        df['rsi'] = rsi_indicator.rsi()
        df['rsi_norm'] = df['rsi'] / 100  # Normalize to 0-1

        # MACD
        macd_indicator = trend.MACD(
            close=df['close'],
            window_fast=config['macd_fast'],
            window_slow=config['macd_slow'],
            window_sign=config['macd_signal']
        )
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()

        # Normalize MACD by price
        df['macd_norm'] = df['macd'] / df['close']
        df['macd_hist_norm'] = df['macd_hist'] / df['close']

        # Bollinger Bands
        bb_indicator = volatility.BollingerBands(
            close=df['close'],
            window=config['bollinger_period'],
            window_dev=config['bollinger_std']
        )
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()

        # Bollinger Band indicators
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume moving averages
        for window in [5, 10, 20]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / (df[f'volume_ma_{window}'] + 1e-10)

        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-10)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures"""
        vol_windows = self.config['features']['volatility_windows']

        # Historical volatility (rolling std of returns)
        for window in vol_windows:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()

        # Parkinson's volatility (uses high-low range)
        for window in [10, 20]:
            hl = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (hl ** 2).rolling(window=window).mean() / (4 * np.log(2))
            )

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features (cyclical encoding)"""
        if 'timestamp' in df.columns:
            # Hour of day (cyclical)
            df['hour'] = df['timestamp'].dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            # Day of week (cyclical)
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            # Weekend indicator
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        return df

    def _add_lag_features(self, df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
        """Add lagged features of key variables"""
        key_features = ['close', 'volume', 'returns', 'rsi_norm']

        for feature in key_features:
            if feature in df.columns:
                for lag in range(1, n_lags + 1):
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

        return df

    def create_multi_horizon_targets(
        self,
        df: pd.DataFrame,
        horizons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create target columns for multiple prediction horizons

        Args:
            df: DataFrame with features
            horizons: List of horizons in minutes (e.g., [30, 60, 180])

        Returns:
            DataFrame with added target columns
        """
        if horizons is None:
            horizons = self.config['model']['horizons']

        logger.info(f"Creating targets for horizons: {horizons}")

        df = df.copy()

        for horizon in horizons:
            # Target: Future price at horizon
            df[f'target_{horizon}min'] = df['close'].shift(-horizon)

            # Target: Percentage change at horizon (more stable for ML)
            df[f'target_{horizon}min_pct'] = (
                (df[f'target_{horizon}min'] - df['close']) / df['close']
            )

        # Drop rows where targets are NaN (at the end)
        df = df.dropna()

        logger.info(f"✓ Created targets for {len(horizons)} horizons")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding OHLCV, timestamp, and targets)

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        exclude_patterns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'target_', 'bb_upper', 'bb_lower', 'bb_middle',  # Derived from features
            'hour', 'day_of_week'  # Raw time features (use sin/cos instead)
        ]

        feature_cols = [
            col for col in df.columns
            if not any(pattern in col for pattern in exclude_patterns)
        ]

        return feature_cols

    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> str:
        """
        Save processed data to CSV

        Args:
            df: DataFrame to save
            filename: Custom filename (optional)

        Returns:
            Path where data was saved
        """
        try:
            if filename is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"features_{timestamp}.csv"

            filepath = self.processed_dir / filename
            df.to_csv(filepath, index=False)

            logger.info(f"✓ Processed data saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise

    def load_latest_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Load the most recent processed data file

        Returns:
            DataFrame with processed data or None if not found
        """
        try:
            # Find latest CSV file
            csv_files = sorted(
                self.processed_dir.glob("*.csv"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            if not csv_files:
                logger.warning("No processed data files found")
                return None

            latest_file = csv_files[0]
            logger.info(f"Loading processed data from {latest_file}")

            df = pd.read_csv(latest_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            logger.info(f"✓ Loaded {len(df)} rows from {latest_file.name}")
            return df

        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            return None

    def process_and_save(self, df: pd.DataFrame) -> Optional[str]:
        """
        Convenience method: Create features, targets, and save

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Path to saved file
        """
        # Create features
        df = self.create_all_features(df)

        # Create multi-horizon targets
        df = self.create_multi_horizon_targets(df)

        # Save
        return self.save_processed_data(df)


def main():
    """CLI entry point for feature engineering"""
    import argparse
    from src.ingestion import DataIngestion

    parser = argparse.ArgumentParser(description="Create features from raw OHLCV data")
    parser.add_argument('--input', type=str, default=None, help='Input CSV file')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')

    args = parser.parse_args()

    # Setup logging
    logger.add("logs/features.log", rotation="1 day", retention="7 days")

    # Load data
    if args.input:
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        logger.info("Loading latest raw data")
        ingestion = DataIngestion(config_path=args.config)
        df = ingestion.load_latest_raw_data()

    if df is None:
        logger.error("No data to process")
        return 1

    # Process features
    engineer = FeatureEngineer(config_path=args.config)
    filepath = engineer.process_and_save(df)

    if filepath:
        print(f"✓ Features saved to: {filepath}")
        return 0
    else:
        print("✗ Feature engineering failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
