"""
Production Prediction Utilities
Helper functions for confidence scoring and fallback predictions
"""

import numpy as np
from config import CONFIDENCE, ETA_BOUNDS, FALLBACK
from utils import haversine_distance
from stops import STOPS

def calculate_confidence(distance_meters, eta_seconds, graph=None):
    """
    Calculate prediction confidence score based on multiple factors.
    
    Args:
        distance_meters: Distance to target stop
        eta_seconds: Predicted ETA
        graph: Optional graph object for additional features
        
    Returns:
        float: Confidence score [0, 1]
    """
    # Base confidence from distance
    confidence = 0.50  # Default
    
    for threshold, conf in sorted(CONFIDENCE['distance_thresholds'].items()):
        if distance_meters < threshold:
            confidence = conf
            break
    
    # Adjust for very short or very long ETAs
    if eta_seconds < 60:  # < 1 minute
        confidence *= 0.9
    elif eta_seconds > 1200:  # > 20 minutes
        confidence *= 0.85
    
    # Adjust for graph quality if available
    if graph is not None and hasattr(graph, 'x'):
        num_nodes = graph.x.shape[0] if len(graph.x.shape) > 1 else 1
        if num_nodes < 3:  # Very few nodes
            confidence *= 0.9
    
    return min(1.0, max(0.0, confidence))

def apply_bounds(prediction_seconds):
    """
    Apply min/max bounds to prediction.
    
    Args:
        prediction_seconds: Raw prediction
        
    Returns:
        tuple: (bounded_prediction, was_capped)
    """
    original = prediction_seconds
    bounded = max(
        ETA_BOUNDS['min_seconds'],
        min(ETA_BOUNDS['max_seconds'], prediction_seconds)
    )
    was_capped = (bounded != original)
    
    return bounded, was_capped

def simple_eta_fallback(bus_lat, bus_lon, target_stop_id):
    """
    Simple distance-based ETA fallback when ML model fails.
    
    Args:
        bus_lat: Bus latitude
        bus_lon: Bus longitude
        target_stop_id: Target stop ID
        
    Returns:
        dict: Prediction result with method='fallback'
    """
    target = STOPS[target_stop_id]
    distance = haversine_distance(
        bus_lat, bus_lon,
        target['lat'], target['lon']
    )
    
    # Assume average speed
    eta_seconds = distance / FALLBACK['avg_speed_ms']
    
    # Apply bounds
    eta_seconds, was_capped = apply_bounds(eta_seconds)
    
    return {
        'eta_seconds': eta_seconds,
        'eta_minutes': eta_seconds / 60,
        'confidence': 0.50,  # Lower confidence for fallback
        'method': 'simple_fallback',
        'was_capped': was_capped,
        'distance_meters': distance
    }

def default_eta_fallback():
    """
    Last resort default ETA when everything fails.
    
    Returns:
        dict: Default prediction
    """
    eta_seconds = FALLBACK['default_eta_minutes'] * 60
    
    return {
        'eta_seconds': eta_seconds,
        'eta_minutes': FALLBACK['default_eta_minutes'],
        'confidence': 0.30,  # Very low confidence
        'method': 'default_fallback',
        'was_capped': False
    }
