"""
Unified Bus ETA Preprocessing Module (Production Version)

This module handles BOTH:
1. Batch preprocessing for model training (historical data)
2. Real-time preprocessing for live predictions (single GPS point)

Key Features:
- Route identification from GPS coordinates
- Real-time compatible features (no 10-min window needed)
- Historical feature lookup (pre-computed)
- New derived features (distance/speed ratio, etc.)

Author: Shreehari Anbazhagan
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Campus boundaries (IITM)
IITM_BOUNDS = {
    'lat_min': 12.9800,
    'lat_max': 13.0200,
    'lon_min': 80.2200,
    'lon_max': 80.2500
}

# Speed thresholds
MAX_SPEED_KMH = 60
MIN_SPEED_MOVING = 1.0

# Trip detection
TRIP_GAP_MINUTES = 15
MIN_TRIP_POINTS = 20

# Training data generation
SAMPLING_INTERVAL_SEC = 10
MAX_ETA_MINUTES = 30

# Bus stops (29 stops across campus)
BUS_STOPS = {
    1: {'name': 'Main Gate', 'lat': 12.9916, 'lon': 80.2336},
    2: {'name': 'Hostel', 'lat': 12.9890, 'lon': 80.2310},
    3: {'name': 'Velachery Gate', 'lat': 12.9850, 'lon': 80.2280},
    4: {'name': 'RP', 'lat': 12.9920, 'lon': 80.2350},
    5: {'name': 'ED', 'lat': 12.9880, 'lon': 80.2320},
    # ... (add remaining 24 stops)
}

# Route definitions (8 routes)
ROUTES = {
    'HOSTEL_MAIN': {
        'stops': [2, 1],
        'name': 'Hostel to Main Gate'
    },
    'MAIN_HOSTEL': {
        'stops': [1, 2],
        'name': 'Main Gate to Hostel'
    },
    'VELACHERY_MAIN': {
        'stops': [3, 1],
        'name': 'Velachery Gate to Main Gate'
    },
    'MAIN_VELACHERY': {
        'stops': [1, 3],
        'name': 'Main Gate to Velachery Gate'
    },
    'VELACHERY_HOSTEL': {
        'stops': [3, 2],
        'name': 'Velachery Gate to Hostel'
    },
    'HOSTEL_VELACHERY': {
        'stops': [2, 3],
        'name': 'Hostel to Velachery Gate'
    },
    'RP_ED': {
        'stops': [4, 5],
        'name': 'RP to ED'
    },
    'ED_RP': {
        'stops': [5, 4],
        'name': 'ED to RP'
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in meters"""
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def is_within_bounds(lat, lon):
    """Check if GPS point is within IITM campus"""
    return (IITM_BOUNDS['lat_min'] <= lat <= IITM_BOUNDS['lat_max'] and
            IITM_BOUNDS['lon_min'] <= lon <= IITM_BOUNDS['lon_max'])


def find_nearest_stop(lat, lon):
    """Find nearest bus stop to given coordinates"""
    min_dist = float('inf')
    nearest_stop = None
    
    for stop_id, stop_info in BUS_STOPS.items():
        dist = haversine_distance(lat, lon, stop_info['lat'], stop_info['lon'])
        if dist < min_dist:
            min_dist = dist
            nearest_stop = stop_id
    
    return nearest_stop, min_dist


def identify_route_from_position(lat, lon, recent_stops=None):
    """
    Identify route from current position and recent stops
    
    For real-time: Use last known stop + current position
    For training: Use stop sequence from trip
    
    Args:
        lat, lon: Current GPS position
        recent_stops: List of recently visited stops (optional)
    
    Returns:
        route_id: Best matching route (or None)
    """
    if recent_stops and len(recent_stops) >= 2:
        # Match based on stop sequence
        for route_id, route_info in ROUTES.items():
            route_stops = route_info['stops']
            # Check if recent stops match route sequence
            if all(stop in route_stops for stop in recent_stops[-2:]):
                return route_id
    
    # Fallback: Find nearest route based on position
    # (Simplified - in production, use route path matching)
    nearest_stop, dist = find_nearest_stop(lat, lon)
    
    # Find routes that include this stop
    for route_id, route_info in ROUTES.items():
        if nearest_stop in route_info['stops']:
            return route_id
    
    return None


# ============================================================================
# CORE PREPROCESSING FUNCTIONS
# ============================================================================

def clean_gps_data(df):
    """Step 1: Clean and validate GPS data"""
    print("\n" + "="*80)
    print("STEP 1: DATA CLEANING")
    print("="*80)
    
    initial_count = len(df)
    
    # Normalize column names (handle case sensitivity)
    df = df.rename(columns={
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Speed': 'speed',
        'DateTime': 'datetime'
    })
    
    # Convert to numeric (handle string values)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    
    # Convert datetime (CRITICAL!)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Drop rows with invalid data
    df = df.dropna(subset=['latitude', 'longitude', 'speed', 'datetime'])
    
    # Remove out-of-bounds points (vectorized!)
    mask = (
        (df['latitude'] >= IITM_BOUNDS['lat_min']) &
        (df['latitude'] <= IITM_BOUNDS['lat_max']) &
        (df['longitude'] >= IITM_BOUNDS['lon_min']) &
        (df['longitude'] <= IITM_BOUNDS['lon_max'])
    )
    df = df[mask]
    
    # Cap unrealistic speeds
    df['speed'] = df['speed'].clip(upper=MAX_SPEED_KMH)
    
    # Sort by bus and time
    df = df.sort_values(['IMEI', 'datetime']).reset_index(drop=True)
    
    print(f"✓ Removed {initial_count - len(df):,} out-of-bounds points")
    print(f"✓ Remaining: {len(df):,} GPS records")
    
    return df


def detect_trips(df):
    """Step 2: Segment GPS data into individual trips"""
    print("\n" + "="*80)
    print("STEP 2: TRIP DETECTION")
    print("="*80)
    
    df['trip_id'] = 0
    trip_counter = 0
    
    for imei in df['IMEI'].unique():
        bus_data = df[df['IMEI'] == imei].copy()
        bus_data['time_gap'] = bus_data['datetime'].diff().dt.total_seconds() / 60
        
        current_trip = trip_counter
        for idx, row in bus_data.iterrows():
            if pd.notna(row['time_gap']) and row['time_gap'] > TRIP_GAP_MINUTES:
                trip_counter += 1
                current_trip = trip_counter
            df.loc[idx, 'trip_id'] = current_trip
        
        trip_counter += 1
    
    # Filter short trips
    trip_lengths = df.groupby('trip_id').size()
    valid_trips = trip_lengths[trip_lengths >= MIN_TRIP_POINTS].index
    df = df[df['trip_id'].isin(valid_trips)].reset_index(drop=True)
    
    print(f"✓ Detected {len(valid_trips)} valid trips")
    
    return df


def assign_routes(df):
    """Step 3: Assign route to each trip"""
    print("\n" + "="*80)
    print("STEP 3: ROUTE MATCHING")
    print("="*80)
    
    df['route_id'] = 'UNKNOWN'
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Matching routes"):
        trip_data = df[df['trip_id'] == trip_id]
        
        # Find stops visited during trip
        stops_visited = []
        for _, row in trip_data.iterrows():
            nearest_stop, dist = find_nearest_stop(row['latitude'], row['longitude'])
            if dist < 100 and nearest_stop not in stops_visited:
                stops_visited.append(nearest_stop)
        
        # Match to route
        for route_id, route_info in ROUTES.items():
            route_stops = route_info['stops']
            matched_stops = [s for s in stops_visited if s in route_stops]
            
            if len(matched_stops) >= min(3, len(route_stops)):
                df.loc[df['trip_id'] == trip_id, 'route_id'] = route_id
                break
    
    matched_trips = (df['route_id'] != 'UNKNOWN').sum() / len(df) * 100
    print(f"✓ Matched {matched_trips:.1f}% of trips to routes")
    
    # Remove unmatched trips
    df = df[df['route_id'] != 'UNKNOWN'].reset_index(drop=True)
    
    return df


def add_distance_features(df):
    """Step 4: Calculate distance features"""
    print("\n" + "="*80)
    print("STEP 4: DISTANCE FEATURES")
    print("="*80)
    
    df['segment_distance_m'] = 0.0
    df['cum_dist_m'] = 0.0
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Calculating distances"):
        mask = df['trip_id'] == trip_id
        trip_data = df[mask].copy()
        
        # Segment distances
        for i in range(1, len(trip_data)):
            prev_idx = trip_data.index[i-1]
            curr_idx = trip_data.index[i]
            
            dist = haversine_distance(
                df.loc[prev_idx, 'latitude'], df.loc[prev_idx, 'longitude'],
                df.loc[curr_idx, 'latitude'], df.loc[curr_idx, 'longitude']
            )
            df.loc[curr_idx, 'segment_distance_m'] = dist
        
        # Cumulative distance
        df.loc[mask, 'cum_dist_m'] = df.loc[mask, 'segment_distance_m'].cumsum()
    
    print(f"✓ Added distance features")
    
    return df


def add_temporal_features(df):
    """Step 5: Add temporal features (real-time compatible)"""
    print("\n" + "="*80)
    print("STEP 5: TEMPORAL FEATURES")
    print("="*80)
    
    # Basic temporal
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = df['hour'].isin([8, 9, 17, 18]).astype(int)
    df['minutes_since_midnight'] = df['hour'] * 60 + df['datetime'].dt.minute
    
    # Cyclical encoding (better for ML)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    print(f"✓ Added temporal features (basic + cyclical)")
    
    return df


def add_speed_dynamics(df):
    """Step 6: Add speed-based features (real-time compatible)"""
    print("\n" + "="*80)
    print("STEP 6: SPEED DYNAMICS")
    print("="*80)
    
    df['acceleration'] = 0.0
    df['speed_ma_1min'] = 0.0
    df['speed_ma_30sec'] = 0.0  # Shorter window for real-time
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Calculating speed dynamics"):
        mask = df['trip_id'] == trip_id
        trip_data = df[mask].copy()
        
        # Acceleration
        df.loc[mask, 'acceleration'] = trip_data['speed'].diff().fillna(0)
        
        # Moving averages (real-time: use smaller windows)
        df.loc[mask, 'speed_ma_30sec'] = trip_data['speed'].rolling(window=30, min_periods=1).mean()
        df.loc[mask, 'speed_ma_1min'] = trip_data['speed'].rolling(window=60, min_periods=1).mean()
    
    print(f"✓ Added speed dynamics")
    
    return df


def add_derived_features(df):
    """
    Step 7: Add NEW derived features (your ideas!)
    
    These are real-time compatible and highly predictive
    """
    print("\n" + "="*80)
    print("STEP 7: DERIVED FEATURES (NEW!)")
    print("="*80)
    
    # 1. Time-to-stop estimate (distance/speed ratio) - YOUR IDEA!
    df['time_to_stop_naive'] = df['dist_to_stop_m'] / (df['speed'] + 0.1)  # Avoid div by zero
    
    # 2. Speed efficiency (actual vs expected)
    df['speed_efficiency'] = df['speed'] / (df['speed_ma_1min'] + 0.1)
    
    # 3. Distance categories (proximity flags)
    df['is_very_close'] = (df['dist_to_stop_m'] < 200).astype(int)
    df['is_close'] = (df['dist_to_stop_m'] < 500).astype(int)
    df['is_far'] = (df['dist_to_stop_m'] > 2000).astype(int)
    
    # 4. Movement state (better than binary is_stopped)
    df['is_moving'] = (df['speed'] >= MIN_SPEED_MOVING).astype(int)
    df['is_slow'] = ((df['speed'] > 0) & (df['speed'] < 5)).astype(int)
    
    # 5. Acceleration state
    df['is_accelerating'] = (df['acceleration'] > 2).astype(int)
    df['is_decelerating'] = (df['acceleration'] < -2).astype(int)
    
    print(f"✓ Added 11 derived features")
    print(f"  - time_to_stop_naive (distance/speed)")
    print(f"  - speed_efficiency")
    print(f"  - Distance categories (3)")
    print(f"  - Movement states (2)")
    print(f"  - Acceleration states (2)")
    
    return df


def find_stop_arrivals(df):
    """Step 8: Identify when buses arrive at stops"""
    print("\n" + "="*80)
    print("STEP 8: STOP ARRIVAL DETECTION")
    print("="*80)
    
    arrivals = []
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Finding arrivals"):
        trip_data = df[df['trip_id'] == trip_id]
        route_id = trip_data['route_id'].iloc[0]
        
        if route_id not in ROUTES:
            continue
        
        route_stops = ROUTES[route_id]['stops']
        
        for stop_id in route_stops:
            stop_info = BUS_STOPS[stop_id]
            
            # Calculate distance to this stop for all points
            distances = trip_data.apply(
                lambda row: haversine_distance(
                    row['latitude'], row['longitude'],
                    stop_info['lat'], stop_info['lon']
                ), axis=1
            )
            
            # Find closest approach
            min_dist_idx = distances.idxmin()
            min_dist = distances[min_dist_idx]
            
            if min_dist < 100:  # Within 100m = arrival
                arrivals.append({
                    'trip_id': trip_id,
                    'route_id': route_id,
                    'stop_id': stop_id,
                    'arrival_time': df.loc[min_dist_idx, 'datetime'],
                    'distance_m': min_dist
                })
    
    arrivals_df = pd.DataFrame(arrivals)
    print(f"✓ Found {len(arrivals_df)} stop arrivals")
    
    return arrivals_df


def generate_training_data(df, arrivals_df):
    """
    Step 9: Generate training examples with labels
    
    For each arrival, sample GPS points before it and calculate ETA
    """
    print("\n" + "="*80)
    print("STEP 9: TRAINING DATA GENERATION")
    print("="*80)
    
    training_rows = []
    
    for _, arrival in tqdm(arrivals_df.iterrows(), total=len(arrivals_df), desc="Generating training rows"):
        trip_id = arrival['trip_id']
        stop_id = arrival['stop_id']
        arrival_time = arrival['arrival_time']
        route_id = arrival['route_id']
        
        # Get trip data before arrival
        trip_data = df[
            (df['trip_id'] == trip_id) &
            (df['datetime'] < arrival_time) &
            (df['datetime'] >= arrival_time - timedelta(minutes=MAX_ETA_MINUTES))
        ].copy()
        
        if len(trip_data) == 0:
            continue
        
        # Sample every N seconds
        trip_data['seconds_from_start'] = (trip_data['datetime'] - trip_data['datetime'].min()).dt.total_seconds()
        trip_data['sample_group'] = (trip_data['seconds_from_start'] // SAMPLING_INTERVAL_SEC).astype(int)
        sampled = trip_data.groupby('sample_group').first().reset_index(drop=True)
        
        # Generate training examples
        for _, row in sampled.iterrows():
            eta_seconds = (arrival_time - row['datetime']).total_seconds()
            
            if eta_seconds <= 0:
                continue
            
            # Calculate distance to target stop
            stop_info = BUS_STOPS[stop_id]
            dist_to_stop = haversine_distance(
                row['latitude'], row['longitude'],
                stop_info['lat'], stop_info['lon']
            )
            
            # Get stop sequence
            route_stops = ROUTES[route_id]['stops']
            stop_sequence = route_stops.index(stop_id) if stop_id in route_stops else -1
            
            training_rows.append({
                # Identifiers
                'trip_id': trip_id,
                'route_id': route_id,
                'stop_id': stop_id,
                'stop_sequence': stop_sequence,
                
                # Location
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                
                # Motion
                'speed': row['speed'],
                'acceleration': row['acceleration'],
                
                # Distance
                'cum_dist_m': row['cum_dist_m'],
                'dist_to_stop_m': dist_to_stop,
                
                # Temporal (basic)
                'hour': row['hour'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'is_peak_hour': row['is_peak_hour'],
                'minutes_since_midnight': row['minutes_since_midnight'],
                
                # Temporal (cyclical)
                'hour_sin': row['hour_sin'],
                'hour_cos': row['hour_cos'],
                'day_sin': row['day_sin'],
                'day_cos': row['day_cos'],
                
                # Speed dynamics
                'speed_ma_30sec': row['speed_ma_30sec'],
                'speed_ma_1min': row['speed_ma_1min'],
                
                # Derived features (NEW!)
                'time_to_stop_naive': row['time_to_stop_naive'],
                'speed_efficiency': row['speed_efficiency'],
                'is_very_close': row['is_very_close'],
                'is_close': row['is_close'],
                'is_far': row['is_far'],
                'is_moving': row['is_moving'],
                'is_slow': row['is_slow'],
                'is_accelerating': row['is_accelerating'],
                'is_decelerating': row['is_decelerating'],
                
                # LABEL
                'ETA_sec': eta_seconds
            })
    
    training_df = pd.DataFrame(training_rows)
    
    print(f"✓ Generated {len(training_df):,} training examples")
    print(f"  Features: {len(training_df.columns) - 1}")
    print(f"  Average ETA: {training_df['ETA_sec'].mean()/60:.1f} minutes")
    
    return training_df


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_data(input_df, output_path='training_data.csv'):
    """
    UNIFIED preprocessing pipeline for batch training
    
    Args:
        input_df: Raw GPS DataFrame from data_loader
        output_path: Path to save processed CSV
    
    Returns:
        Training DataFrame ready for XGBoost
    """
    print("\n" + "="*80)
    print("BUS ETA PREDICTION - UNIFIED PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Execute pipeline
    df = clean_gps_data(input_df)
    df = detect_trips(df)
    df = assign_routes(df)
    df = add_distance_features(df)
    df = add_temporal_features(df)
    df = add_speed_dynamics(df)
    
    # Find arrivals (need dist_to_stop_m first)
    arrivals_df = find_stop_arrivals(df)
    
    # Add dist_to_stop_m to df (for derived features)
    # This is a placeholder - will be calculated per training example
    df['dist_to_stop_m'] = 0.0
    
    # Add derived features
    df = add_derived_features(df)
    
    # Generate training data
    training_df = generate_training_data(df, arrivals_df)
    
    # Save
    print("\n" + "="*80)
    print("SAVING DATA")
    print("="*80)
    
    training_df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Summary
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Total training examples: {len(training_df):,}")
    print(f"Unique trips: {training_df['trip_id'].nunique():,}")
    print(f"Unique stops: {training_df['stop_id'].nunique()}")
    print(f"Routes covered: {training_df['route_id'].nunique()}")
    print(f"Features: {len(training_df.columns) - 1}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return training_df


# ============================================================================
# REAL-TIME PREDICTION PREPROCESSING
# ============================================================================

def preprocess_realtime(gps_point, target_stop_id, route_id=None):
    """
    Preprocess SINGLE GPS point for real-time prediction
    
    NO sliding window needed - works with just current point!
    
    Args:
        gps_point: Dict with keys: lat, lon, speed, timestamp
        target_stop_id: Which stop to predict ETA for
        route_id: Optional route (will auto-detect if None)
    
    Returns:
        Feature dict ready for model.predict()
    
    Example:
        >>> point = {
        ...     'lat': 12.9916,
        ...     'lon': 80.2336,
        ...     'speed': 25.5,
        ...     'timestamp': datetime.now()
        ... }
        >>> features = preprocess_realtime(point, target_stop_id=2)
        >>> eta_sec = model.predict([features])[0]
    """
    # Extract from GPS point
    lat = gps_point['lat']
    lon = gps_point['lon']
    speed = gps_point['speed']
    timestamp = gps_point['timestamp']
    
    # Auto-detect route if not provided
    if route_id is None:
        route_id = identify_route_from_position(lat, lon)
        if route_id is None:
            raise ValueError("Cannot identify route from position")
    
    # Calculate distance to target stop
    stop_info = BUS_STOPS[target_stop_id]
    dist_to_stop = haversine_distance(lat, lon, stop_info['lat'], stop_info['lon'])
    
    # Get stop sequence
    route_stops = ROUTES[route_id]['stops']
    stop_sequence = route_stops.index(target_stop_id) if target_stop_id in route_stops else -1
    
    # Temporal features
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if hour in [8, 9, 17, 18] else 0
    minutes_since_midnight = hour * 60 + timestamp.minute
    
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Derived features (real-time compatible!)
    time_to_stop_naive = dist_to_stop / (speed + 0.1)
    is_very_close = 1 if dist_to_stop < 200 else 0
    is_close = 1 if dist_to_stop < 500 else 0
    is_far = 1 if dist_to_stop > 2000 else 0
    is_moving = 1 if speed >= MIN_SPEED_MOVING else 0
    is_slow = 1 if 0 < speed < 5 else 0
    
    # Build feature dict
    features = {
        # Identifiers
        'route_id': route_id,
        'stop_id': target_stop_id,
        'stop_sequence': stop_sequence,
        
        # Location
        'latitude': lat,
        'longitude': lon,
        
        # Motion (current only - no history needed!)
        'speed': speed,
        'acceleration': 0.0,  # Unknown without history
        
        # Distance
        'cum_dist_m': 0.0,  # Unknown without trip start
        'dist_to_stop_m': dist_to_stop,
        
        # Temporal
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour,
        'minutes_since_midnight': minutes_since_midnight,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
        
        # Speed dynamics (use current speed as proxy)
        'speed_ma_30sec': speed,
        'speed_ma_1min': speed,
        
        # Derived features
        'time_to_stop_naive': time_to_stop_naive,
        'speed_efficiency': 1.0,  # Unknown without history
        'is_very_close': is_very_close,
        'is_close': is_close,
        'is_far': is_far,
        'is_moving': is_moving,
        'is_slow': is_slow,
        'is_accelerating': 0,  # Unknown without history
        'is_decelerating': 0,  # Unknown without history
    }
    
    return features


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Batch preprocessing for training
    from data_loader import load_multiple_days
    
    print("="*80)
    print("EXAMPLE 1: BATCH PREPROCESSING")
    print("="*80)
    
    dates = ['20251103', '20251104', '20251105']
    df = load_multiple_days(dates)
    training_data = preprocess_data(df, output_path='training_data.csv')
    
    print("\n" + "="*80)
    print("EXAMPLE 2: REAL-TIME PREPROCESSING")
    print("="*80)
    
    # Simulate real-time GPS point
    gps_point = {
        'lat': 12.9916,
        'lon': 80.2336,
        'speed': 25.5,
        'timestamp': datetime.now()
    }
    
    # Preprocess for prediction
    features = preprocess_realtime(gps_point, target_stop_id=2, route_id='MAIN_HOSTEL')
    
    print(f"\n✓ Real-time features generated:")
    print(f"  Distance to stop: {features['dist_to_stop_m']:.0f}m")
    print(f"  Naive ETA: {features['time_to_stop_naive']/60:.1f} min")
    print(f"  Route: {features['route_id']}")
    print(f"\n✓ Ready for model.predict()!")
