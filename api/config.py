"""
Production Configuration
Centralized configuration for validation, bounds, and thresholds
"""

# GPS Validation Bounds
GPS_VALIDATION = {
    # Global bounds
    'lat_min': -90,
    'lat_max': 90,
    'lon_min': -180,
    'lon_max': 180,
    
    # Chennai/IIT Madras area bounds (with buffer)
    'chennai_lat_min': 12.85,
    'chennai_lat_max': 13.15,
    'chennai_lon_min': 80.10,
    'chennai_lon_max': 80.35,
    
    # Speed bounds (km/h)
    'speed_min': 0,
    'speed_max': 150,
}

# ETA Prediction Bounds
ETA_BOUNDS = {
    'min_seconds': 30,      # 30 seconds minimum
    'max_seconds': 1800,    # 30 minutes maximum
    'min_minutes': 0.5,
    'max_minutes': 30,
}

# Confidence Score Thresholds
CONFIDENCE = {
    'distance_thresholds': {
        500: 0.95,   # < 500m = 95% confidence
        1000: 0.85,  # < 1km = 85% confidence
        2000: 0.75,  # < 2km = 75% confidence
        5000: 0.60,  # < 5km = 60% confidence
    },
    'low_confidence_threshold': 0.60,
}

# Fallback Configuration
FALLBACK = {
    'avg_speed_kmh': 20,     # Average city speed
    'avg_speed_ms': 5.56,    # 20 km/h in m/s
    'default_eta_minutes': 15,  # Last resort default
}

# Model Configuration
MODEL = {
    'use_gnn': False,  # GNN weight is 0, disabled by default
    'model_dir': '.',
}

# Route Matching
ROUTE_MATCHING = {
    'tolerance_meters': 100,  # Increased from 50m
    'min_match_percentage': 0.5,
}

# Data Quality
DATA_QUALITY = {
    'min_trip_points': 10,
    'max_speed_jump_kmh': 50,  # Max speed change between points
    'max_time_gap_seconds': 300,  # 5 minutes
}

# Monitoring
MONITORING = {
    'log_all_predictions': True,
    'alert_on_high_eta': True,
    'alert_on_low_confidence': True,
    'alert_threshold_eta_minutes': 25,
}
