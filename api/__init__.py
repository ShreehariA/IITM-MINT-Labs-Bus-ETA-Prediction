"""
IIT Madras Bus ETA Prediction API
Production-ready real-time prediction system using graph-based ensemble models
"""

__version__ = "1.0.0"

from .feature_extractor import LiveDataPreprocessor
from .predictor import EnsembleModel
from .stops import STOPS, ROUTES, get_stop_info, get_route_stops, get_upcoming_stops

__all__ = [
    'LiveDataPreprocessor',
    'EnsembleModel',
    'STOPS',
    'ROUTES',
    'get_stop_info',
    'get_route_stops',
    'get_upcoming_stops'
]
