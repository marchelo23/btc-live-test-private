"""
BTC MLOps Local - Source Package
Production-ready Bitcoin price prediction system
"""

__version__ = "1.0.0"
__author__ = "BTC MLOps Team"

from . import ingestion
from . import features
from . import train
from . import inference

__all__ = ["ingestion", "features", "train", "inference"]
