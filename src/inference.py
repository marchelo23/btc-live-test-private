"""
Inference Module
Loads trained models and makes predictions
Optimized for fast inference on Parrot OS
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from loguru import logger
from typing import Dict, List, Optional
from datetime import datetime


class ModelInference:
    """
    Handles model loading and prediction
    Caches models in memory for fast inference
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize inference engine

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models_dir = Path(self.config['model']['models_dir'])

        self.models = {}  # Cache for loaded models
        self.model_timestamps = {}

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def load_model(self, horizon: int, force_reload: bool = False) -> Optional[xgb.XGBRegressor]:
        """
        Load model for a specific horizon

        Args:
            horizon: Prediction horizon in minutes
            force_reload: Force reload from disk

        Returns:
            Loaded XGBoost model or None if not found
        """
        # Check cache first
        if horizon in self.models and not force_reload:
            logger.debug(f"Using cached model for {horizon}min")
            return self.models[horizon]

        try:
            prefix = self.config['model']['model_prefix']

            # Look for latest model (symlink)
            latest_path = self.models_dir / f"{prefix}_{horizon}min_latest.json"

            if latest_path.exists():
                model_path = latest_path
            else:
                # Find most recent model file
                model_files = sorted(
                    self.models_dir.glob(f"{prefix}_{horizon}min_*.json"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

                if not model_files:
                    logger.error(f"No model found for {horizon}min")
                    return None

                model_path = model_files[0]

            # Load model
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))

            # Cache model
            self.models[horizon] = model
            self.model_timestamps[horizon] = datetime.now()

            logger.info(f"✓ Loaded model for {horizon}min from {model_path.name}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model for {horizon}min: {e}")
            return None

    def load_all_models(self) -> Dict[int, xgb.XGBRegressor]:
        """
        Load all configured models

        Returns:
            Dictionary mapping horizon to model
        """
        horizons = self.config['model']['horizons']
        logger.info(f"Loading models for {len(horizons)} horizons")

        loaded_models = {}

        for horizon in horizons:
            model = self.load_model(horizon)
            if model is not None:
                loaded_models[horizon] = model

        logger.info(f"✓ Loaded {len(loaded_models)}/{len(horizons)} models")

        return loaded_models

    def predict_single_horizon(
        self,
        features: pd.DataFrame,
        horizon: int
    ) -> Optional[Dict[str, float]]:
        """
        Make prediction for a single horizon

        Args:
            features: DataFrame with feature columns
            horizon: Prediction horizon in minutes

        Returns:
            Dictionary with prediction results
        """
        try:
            # Load model
            model = self.load_model(horizon)
            if model is None:
                return None

            # Get feature columns from model
            # Extract only feature columns (exclude 'close')
            feature_cols = [col for col in features.columns if col != 'close']
            X = features[feature_cols].values

            # Make prediction
            prediction_pct = float(model.predict(X)[0])

            # Get current price (last close price in features)
            if 'close' in features.columns:
                current_price = float(features['close'].iloc[-1])
            else:
                current_price = None

            # Calculate predicted price
            if current_price is not None:
                predicted_price = current_price * (1 + prediction_pct)
            else:
                predicted_price = None

            # Determine signal
            if prediction_pct > 0.002:  # > 0.2% increase
                signal = "BUY"
            elif prediction_pct < -0.002:  # > 0.2% decrease
                signal = "SELL"
            else:
                signal = "HOLD"

            # Confidence score (based on prediction magnitude)
            confidence = min(1.0, abs(prediction_pct) * 100)

            result = {
                'horizon_min': horizon,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change_pct': prediction_pct * 100,
                'signal': signal,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Prediction failed for {horizon}min: {e}")
            return None

    def predict_all_horizons(
        self,
        features: pd.DataFrame
    ) -> Dict[int, Dict[str, float]]:
        """
        Make predictions for all horizons

        Args:
            features: DataFrame with feature columns

        Returns:
            Dictionary mapping horizon to prediction results
        """
        horizons = self.config['model']['horizons']
        predictions = {}

        for horizon in horizons:
            result = self.predict_single_horizon(features, horizon)
            if result is not None:
                predictions[horizon] = result

        return predictions

    def prepare_features_from_raw(
        self,
        df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Prepare features from raw OHLCV data

        Args:
            df: DataFrame with raw OHLCV data

        Returns:
            DataFrame with features ready for prediction (includes 'close' column)
        """
        try:
            from src.features import FeatureEngineer

            # Create features
            engineer = FeatureEngineer()
            df_features = engineer.create_all_features(df)

            # Get feature columns
            feature_cols = engineer.get_feature_columns(df_features)

            # Return feature columns + close price (last row)
            # We need close for calculating predicted price
            cols_to_return = feature_cols + ['close']
            return df_features[cols_to_return].iloc[[-1]]

        except Exception as e:
            logger.error(f"Failed to prepare features: {e}")
            return None

    def predict_from_raw_data(
        self,
        df: pd.DataFrame
    ) -> Dict[int, Dict[str, float]]:
        """
        End-to-end prediction from raw OHLCV data

        Args:
            df: DataFrame with raw OHLCV data

        Returns:
            Dictionary of predictions for all horizons
        """
        # Prepare features
        features = self.prepare_features_from_raw(df)

        if features is None:
            logger.error("Failed to prepare features")
            return {}

        # Make predictions
        return self.predict_all_horizons(features)

    def get_loaded_models_info(self) -> Dict[str, any]:
        """
        Get information about loaded models

        Returns:
            Dictionary with model information
        """
        info = {
            'loaded_models': list(self.models.keys()),
            'total_models': len(self.models),
            'load_timestamps': {
                horizon: timestamp.isoformat()
                for horizon, timestamp in self.model_timestamps.items()
            }
        }

        return info


def main():
    """CLI entry point for inference"""
    import argparse
    from src.ingestion import DataIngestion

    parser = argparse.ArgumentParser(description="Make BTC price predictions")
    parser.add_argument('--horizon', type=int, default=None, help='Predict specific horizon')
    parser.add_argument('--all', action='store_true', help='Predict all horizons')
    parser.add_argument('--fetch', action='store_true', help='Fetch latest data before predicting')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')

    args = parser.parse_args()

    # Setup logging
    logger.add("logs/inference.log", rotation="1 day", retention="7 days")

    # Initialize inference
    inference = ModelInference(config_path=args.config)

    # Get data
    if args.fetch:
        logger.info("Fetching latest data...")
        ingestion = DataIngestion(config_path=args.config)
        df = ingestion.fetch_ohlcv(lookback_days=2)  # Just need recent data
    else:
        logger.info("Loading latest raw data...")
        ingestion = DataIngestion(config_path=args.config)
        df = ingestion.load_latest_raw_data()

    if df is None:
        logger.error("No data available for prediction")
        return 1

    # Make predictions
    if args.all:
        # Predict all horizons
        predictions = inference.predict_from_raw_data(df)

        if predictions:
            print("\n" + "=" * 80)
            print("BTC PRICE PREDICTIONS")
            print("=" * 80)

            for horizon, result in sorted(predictions.items()):
                print(f"\n{horizon}-Minute Prediction:")
                print(f"  Current Price: ${result['current_price']:,.2f}")
                print(f"  Predicted Price: ${result['predicted_price']:,.2f}")
                print(f"  Change: {result['predicted_change_pct']:+.2f}%")
                print(f"  Signal: {result['signal']}")
                print(f"  Confidence: {result['confidence']:.2f}")

            print("\n" + "=" * 80)
        else:
            print("✗ Prediction failed")
            return 1

    elif args.horizon:
        # Predict specific horizon
        features = inference.prepare_features_from_raw(df)
        result = inference.predict_single_horizon(features, args.horizon)

        if result:
            print(f"\n{args.horizon}-Minute Prediction:")
            print(f"  Current Price: ${result['current_price']:,.2f}")
            print(f"  Predicted Price: ${result['predicted_price']:,.2f}")
            print(f"  Change: {result['predicted_change_pct']:+.2f}%")
            print(f"  Signal: {result['signal']}")
        else:
            print("✗ Prediction failed")
            return 1

    else:
        logger.error("Specify --horizon or --all")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
