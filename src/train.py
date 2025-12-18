"""
Model Training Module
Trains XGBoost models for multi-horizon BTC price prediction
Optimized for local CPU training on Parrot OS
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
from typing import Dict, List, Optional, Tuple


class ModelTrainer:
    """
    Trains XGBoost models for Bitcoin price prediction
    Supports multi-horizon training
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize model trainer

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models_dir = Path(self.config['model']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.scalers = {}
        self.metrics = {}

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training

        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Extract features and target
        X = df[feature_cols].values
        y = df[target_col].values

        # Time-based split (important for time series!)
        test_size = self.config['model']['test_size']

        # Don't shuffle for time series
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            shuffle=False  # Critical for time series
        )

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        horizon: int
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model for a specific horizon

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            horizon: Prediction horizon in minutes

        Returns:
            Trained XGBoost model
        """
        logger.info(f"Training model for {horizon}min horizon...")

        # Get XGBoost parameters from config
        xgb_config = self.config['model']['xgboost']

        # Create model with early_stopping_rounds in constructor
        model = xgb.XGBRegressor(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            subsample=xgb_config['subsample'],
            colsample_bytree=xgb_config['colsample_bytree'],
            min_child_weight=xgb_config['min_child_weight'],
            gamma=xgb_config['gamma'],
            reg_alpha=xgb_config['reg_alpha'],
            reg_lambda=xgb_config['reg_lambda'],
            random_state=xgb_config['random_state'],
            tree_method='hist',  # Faster on CPU
            verbosity=1,
            early_stopping_rounds=20
        )

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        logger.info(f"✓ Model trained. Best iteration: {model.best_iteration}")

        return model

    def evaluate_model(
        self,
        model: xgb.XGBRegressor,
        X_test: np.ndarray,
        y_test: np.ndarray,
        horizon: int
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            horizon: Prediction horizon

        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'horizon_min': horizon,
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        }

        # Directional accuracy
        y_test_direction = np.sign(y_test)
        y_pred_direction = np.sign(y_pred)
        metrics['directional_accuracy'] = (y_test_direction == y_pred_direction).mean()

        logger.info(f"Metrics for {horizon}min:")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")

        return metrics

    def save_model(
        self,
        model: xgb.XGBRegressor,
        horizon: int,
        metrics: Dict[str, float]
    ) -> str:
        """
        Save model to disk

        Args:
            model: Trained model
            horizon: Prediction horizon
            metrics: Model metrics

        Returns:
            Path where model was saved
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = self.config['model']['model_prefix']

            # Model filename
            model_filename = f"{prefix}_{horizon}min_{timestamp}.json"
            model_path = self.models_dir / model_filename

            # Save model (JSON format for XGBoost)
            model.save_model(str(model_path))

            # Save metrics
            metrics_filename = f"{prefix}_{horizon}min_{timestamp}_metrics.json"
            metrics_path = self.models_dir / metrics_filename

            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            # Create symlink to latest model
            latest_link = self.models_dir / f"{prefix}_{horizon}min_latest.json"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(model_filename)

            logger.info(f"✓ Model saved to {model_path}")
            logger.info(f"✓ Metrics saved to {metrics_path}")

            return str(model_path)

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def train_single_horizon(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        horizon: int
    ) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
        """
        Train model for a single horizon

        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature columns
            horizon: Prediction horizon in minutes

        Returns:
            Tuple of (model, metrics)
        """
        logger.info(f"=" * 80)
        logger.info(f"Training {horizon}-minute prediction model")
        logger.info(f"=" * 80)

        # Target column
        target_col = f'target_{horizon}min_pct'

        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in DataFrame")

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, feature_cols, target_col)

        # Train model
        model = self.train_model(X_train, y_train, X_test, y_test, horizon)

        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test, horizon)

        # Save model
        self.save_model(model, horizon, metrics)

        # Store in memory
        self.models[horizon] = model
        self.metrics[horizon] = metrics

        return model, metrics

    def train_all_horizons(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[int, Tuple[xgb.XGBRegressor, Dict[str, float]]]:
        """
        Train models for all configured horizons

        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature columns

        Returns:
            Dictionary mapping horizon to (model, metrics)
        """
        horizons = self.config['model']['horizons']
        logger.info(f"Training models for {len(horizons)} horizons: {horizons}")

        results = {}

        for horizon in horizons:
            try:
                model, metrics = self.train_single_horizon(df, feature_cols, horizon)
                results[horizon] = (model, metrics)

            except Exception as e:
                logger.error(f"Failed to train model for {horizon}min: {e}")
                continue

        # Save summary
        self._save_training_summary(results)

        logger.info("=" * 80)
        logger.info("✓ All models trained successfully!")
        logger.info("=" * 80)

        return results

    def _save_training_summary(self, results: Dict[int, Tuple]) -> None:
        """Save summary of all training results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = self.models_dir / f"training_summary_{timestamp}.json"

            summary = {
                'timestamp': timestamp,
                'horizons': {},
                'overall_metrics': {}
            }

            for horizon, (model, metrics) in results.items():
                summary['horizons'][f'{horizon}min'] = metrics

            # Calculate average metrics
            all_metrics = [metrics for _, metrics in results.values()]
            if all_metrics:
                summary['overall_metrics'] = {
                    'avg_rmse': np.mean([m['rmse'] for m in all_metrics]),
                    'avg_mae': np.mean([m['mae'] for m in all_metrics]),
                    'avg_r2': np.mean([m['r2'] for m in all_metrics]),
                    'avg_directional_accuracy': np.mean([m['directional_accuracy'] for m in all_metrics])
                }

            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"✓ Training summary saved to {summary_path}")

        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")


def main():
    """CLI entry point for model training"""
    import argparse
    from src.features import FeatureEngineer

    parser = argparse.ArgumentParser(description="Train BTC prediction models")
    parser.add_argument('--horizon', type=int, default=None, help='Train specific horizon (e.g., 30, 60)')
    parser.add_argument('--all', action='store_true', help='Train all horizons')
    parser.add_argument('--input', type=str, default=None, help='Input CSV file with features')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')

    args = parser.parse_args()

    # Setup logging
    logger.add("logs/train.log", rotation="1 day", retention="7 days")

    # Load data
    if args.input:
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        logger.info("Loading latest processed data")
        engineer = FeatureEngineer(config_path=args.config)
        df = engineer.load_latest_processed_data()

    if df is None:
        logger.error("No data to train on")
        return 1

    # Get feature columns
    engineer = FeatureEngineer(config_path=args.config)
    feature_cols = engineer.get_feature_columns(df)

    logger.info(f"Using {len(feature_cols)} features for training")

    # Initialize trainer
    trainer = ModelTrainer(config_path=args.config)

    # Train
    if args.all:
        # Train all horizons
        results = trainer.train_all_horizons(df, feature_cols)
        print(f"✓ Trained {len(results)} models successfully")

    elif args.horizon:
        # Train specific horizon
        model, metrics = trainer.train_single_horizon(df, feature_cols, args.horizon)
        print(f"✓ Model for {args.horizon}min trained successfully")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")

    else:
        logger.error("Specify --horizon or --all")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
