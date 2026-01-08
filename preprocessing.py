"""
Bus ETA Prediction - Data Preprocessing Pipeline

This module handles all data cleaning, filtering, trip detection, route matching,
and feature engineering to prepare GPS data for model training.

Based on EDA findings:
- Remove out-of-bounds data (~2% of records)
- Cap speeds at 60 km/h
- Detect trips using 15-minute gap threshold
- Match trips to 8 predefined routes
- Generate training examples with ~25 features

Author: Shreehari Anbazhagan
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# IITM Campus Boundaries (from EDA)
IITM_BOUNDS = {
    'lat_min': 12.98,
    'lat_max': 13.06,
    'lon_min': 80.22,
    'lon_max': 80.29
}

# Speed thresholds
MAX_SPEED_KMH = 60  # Cap unrealistic speeds
MIN_SPEED_MOVING = 1  # Threshold for "stopped"

# Trip detection
TRIP_GAP_MINUTES = 15  # Gap to detect new trip
MIN_TRIP_POINTS = 20  # Minimum points for valid trip

# Sampling for training data
SAMPLING_INTERVAL_SEC = 10  # Sample every 10 seconds
MAX_ETA_MINUTES = 30  # Only use points within 30 min of arrival

# Bus stops (29 stops from your data)
BUS_STOPS = {
    1: {'name': 'Velachery Gate', 'lat': 12.988763557585395, 'lon': 80.22367960515129},
    2: {'name': 'NAC2 → GC', 'lat': 12.990236, 'lon': 80.227548},
    3: {'name': 'NAC2 → Velachery', 'lat': 12.9898636, 'lon': 80.2271109},
    4: {'name': 'CRC → GC', 'lat': 12.99089, 'lon': 80.230274},
    5: {'name': 'CRC → Velachery', 'lat': 12.99079680, 'lon': 80.23016660},
    6: {'name': 'HSB → Velachery', 'lat': 12.99076780, 'lon': 80.23168950},
    7: {'name': 'HSB → GC', 'lat': 12.991037, 'lon': 80.232065},
    8: {'name': 'KV → Main Gate', 'lat': 12.991868573807471, 'lon': 80.23367454352295},
    9: {'name': 'KV → GC', 'lat': 12.991910, 'lon': 80.233785},
    10: {'name': 'Post Office → Main', 'lat': 12.99388680, 'lon': 80.23428760},
    11: {'name': 'Post Office → GC', 'lat': 12.99393340, 'lon': 80.23459350},
    12: {'name': 'E1 → Main Gate', 'lat': 12.9960173, 'lon': 80.2359173},
    13: {'name': 'E1 → GC', 'lat': 12.996117, 'lon': 80.236184},
    14: {'name': 'Vana Vani → Main', 'lat': 12.998704, 'lon': 80.2391799},
    15: {'name': 'Vana Vani → GC', 'lat': 12.999084, 'lon': 80.239380},
    16: {'name': 'D1 → Main Gate', 'lat': 13.002546, 'lon': 80.240091},
    17: {'name': 'D1 → GC', 'lat': 13.002679, 'lon': 80.240219},
    18: {'name': 'Main Gate', 'lat': 13.00612964850378, 'lon': 80.24191299117685},
    19: {'name': 'GC → Hostel', 'lat': 12.991342, 'lon': 80.233637},
    20: {'name': 'Library', 'lat': 12.9907517, 'lon': 80.2334554},
    21: {'name': 'OAT → Hostel', 'lat': 12.989257, 'lon': 80.233031},
    22: {'name': 'Gymkhana → GC', 'lat': 12.9866372, 'lon': 80.2332951},
    23: {'name': 'Gymkhana → Hostel', 'lat': 12.986615, 'lon': 80.233366},
    24: {'name': 'Narmada → GC', 'lat': 12.9862759, 'lon': 80.2350107},
    25: {'name': 'Narmada → Hostel', 'lat': 12.986546, 'lon': 80.235301},
    26: {'name': 'Jamuna/Ganga (Hostel)', 'lat': 12.986566332825637, 'lon': 80.23855704439099},
    28: {'name': 'RP Stand', 'lat': 12.990088, 'lon': 80.241799},
    29: {'name': 'ED Stand', 'lat': 12.989874633072303, 'lon': 80.22646042038593},
}

# Routes (8 routes from your data)
ROUTES = {
    'HOSTEL_MAIN': {'name': 'Hostel → Main Gate', 'stops': [26, 24, 22, 20, 6, 5, 3, 1, 2, 4, 7, 8, 10, 12, 14, 16, 18]},
    'MAIN_HOSTEL': {'name': 'Main Gate → Hostel', 'stops': [18, 17, 15, 13, 11, 9, 6, 5, 3, 1, 2, 4, 7, 19, 21, 23, 25, 26]},
    'VG_MAIN': {'name': 'Velachery → Main Gate', 'stops': [1, 2, 4, 7, 8, 10, 12, 14, 16, 18]},
    'VG_HOSTEL': {'name': 'Velachery → Hostel', 'stops': [1, 2, 4, 7, 19, 21, 23, 25, 26]},
    'MAIN_VG': {'name': 'Main Gate → Velachery', 'stops': [18, 17, 15, 13, 11, 9, 6, 5, 3, 1]},
    'HOSTEL_VG': {'name': 'Hostel → Velachery', 'stops': [26, 24, 22, 20, 6, 5, 3, 1]},
    'RP_ED': {'name': 'RP → ED', 'stops': [28, 22, 21, 20, 6, 5, 3, 29]},
    'ED_RP': {'name': 'ED → RP', 'stops': [29, 2, 4, 7, 19, 21, 23, 28]},
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance in meters between two GPS points using Haversine formula
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def is_within_bounds(lat, lon):
    """Check if GPS point is within IITM campus bounds"""
    return (IITM_BOUNDS['lat_min'] <= lat <= IITM_BOUNDS['lat_max'] and
            IITM_BOUNDS['lon_min'] <= lon <= IITM_BOUNDS['lon_max'])


def find_nearest_stop(lat, lon, max_distance=100):
    """
    Find the nearest bus stop to given coordinates
    
    Args:
        lat, lon: GPS coordinates
        max_distance: Maximum distance in meters to consider
    
    Returns:
        (stop_id, distance) or (None, None) if no stop within max_distance
    """
    min_dist = float('inf')
    nearest_stop = None
    
    for stop_id, stop_info in BUS_STOPS.items():
        dist = haversine_distance(lat, lon, stop_info['lat'], stop_info['lon'])
        if dist < min_dist:
            min_dist = dist
            nearest_stop = stop_id
    
    if min_dist <= max_distance:
        return nearest_stop, min_dist
    return None, None


# ============================================================================
# STEP 1: DATA CLEANING
# ============================================================================

def clean_gps_data(df):
    """
    Clean GPS data based on EDA findings:
    - Remove out-of-bounds points
    - Cap unrealistic speeds
    - Remove extreme transmission delays
    
    Args:
        df: Raw GPS DataFrame
    
    Returns:
        Cleaned DataFrame with statistics
    """
    print("\n" + "="*80)
    print("STEP 1: DATA CLEANING")
    print("="*80)
    
    initial_count = len(df)
    print(f"\nInitial records: {initial_count:,}")
    
    # 1. Remove out-of-bounds points
    df['within_bounds'] = df.apply(
        lambda row: is_within_bounds(row['Latitude'], row['Longitude']), 
        axis=1
    )
    out_of_bounds = (~df['within_bounds']).sum()
    df = df[df['within_bounds']].drop('within_bounds', axis=1).copy()
    print(f"✓ Removed {out_of_bounds:,} out-of-bounds points ({out_of_bounds/initial_count*100:.2f}%)")
    
    # 2. Cap unrealistic speeds
    high_speed_count = (df['Speed'] > MAX_SPEED_KMH).sum()
    df.loc[df['Speed'] > MAX_SPEED_KMH, 'Speed'] = MAX_SPEED_KMH
    print(f"✓ Capped {high_speed_count:,} speeds above {MAX_SPEED_KMH} km/h ({high_speed_count/initial_count*100:.2f}%)")
    
    # 3. Ensure DateTime is parsed
    if not pd.api.types.is_datetime64_any_dtype(df['DateTime']):
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # 4. Sort by IMEI and DateTime
    df = df.sort_values(['IMEI', 'DateTime']).reset_index(drop=True)
    
    final_count = len(df)
    removed = initial_count - final_count
    print(f"\nFinal records: {final_count:,}")
    print(f"Total removed: {removed:,} ({removed/initial_count*100:.2f}%)")
    
    return df


# ============================================================================
# STEP 2: TRIP DETECTION
# ============================================================================

def detect_trips(df):
    """
    Segment GPS data into individual trips based on time gaps
    
    Args:
        df: Cleaned GPS DataFrame
    
    Returns:
        DataFrame with trip_id column added
    """
    print("\n" + "="*80)
    print("STEP 2: TRIP DETECTION")
    print("="*80)
    
    df = df.copy()
    df['trip_id'] = 0
    trip_counter = 0
    
    for imei in tqdm(df['IMEI'].unique(), desc="Detecting trips"):
        mask = df['IMEI'] == imei
        imei_data = df[mask].copy()
        
        # Calculate time gaps
        imei_data['time_gap_min'] = imei_data['DateTime'].diff().dt.total_seconds() / 60
        
        # Mark new trips (gap > threshold or first record)
        imei_data['is_new_trip'] = (
            (imei_data['time_gap_min'] > TRIP_GAP_MINUTES) | 
            (imei_data['time_gap_min'].isna())
        )
        imei_data['trip_num'] = imei_data['is_new_trip'].cumsum()
        
        # Assign unique trip IDs
        df.loc[mask, 'trip_id'] = trip_counter + imei_data['trip_num']
        trip_counter = df.loc[mask, 'trip_id'].max() + 1
    
    # Remove very short trips
    trip_lengths = df.groupby('trip_id').size()
    valid_trips = trip_lengths[trip_lengths >= MIN_TRIP_POINTS].index
    df = df[df['trip_id'].isin(valid_trips)].copy()
    
    print(f"✓ Detected {len(valid_trips):,} valid trips")
    print(f"✓ Removed {len(trip_lengths) - len(valid_trips):,} short trips (<{MIN_TRIP_POINTS} points)")
    print(f"✓ Average trip length: {trip_lengths[valid_trips].mean():.0f} GPS points")
    
    return df


# ============================================================================
# STEP 3: ROUTE MATCHING
# ============================================================================

def match_trip_to_route(trip_df, min_stops_matched=3):
    """
    Match a trip to a route based on stops visited
    
    Args:
        trip_df: DataFrame for a single trip
        min_stops_matched: Minimum stops to match for valid route
    
    Returns:
        (route_code, match_score, stops_visited)
    """
    # Find stops visited by this trip
    trip_stops = []
    for _, row in trip_df.iterrows():
        stop_id, dist = find_nearest_stop(row['Latitude'], row['Longitude'])
        if stop_id and stop_id not in trip_stops:
            trip_stops.append(stop_id)
    
    # Match to routes
    best_route = None
    best_score = 0
    
    for route_code, route_info in ROUTES.items():
        route_stops = route_info['stops']
        # Count sequential matches
        matches = sum(1 for stop in trip_stops if stop in route_stops)
        
        if matches > best_score and matches >= min_stops_matched:
            best_score = matches
            best_route = route_code
    
    return best_route, best_score, trip_stops


def assign_routes(df):
    """
    Assign route IDs to all trips
    
    Args:
        df: DataFrame with trip_id column
    
    Returns:
        DataFrame with route_id column added
    """
    print("\n" + "="*80)
    print("STEP 3: ROUTE MATCHING")
    print("="*80)
    
    df = df.copy()
    df['route_id'] = None
    
    trip_ids = df['trip_id'].unique()
    matched_count = 0
    
    for trip_id in tqdm(trip_ids, desc="Matching routes"):
        trip_df = df[df['trip_id'] == trip_id]
        route_code, score, stops = match_trip_to_route(trip_df)
        
        if route_code:
            df.loc[df['trip_id'] == trip_id, 'route_id'] = route_code
            matched_count += 1
    
    # Remove trips without route match
    df = df[df['route_id'].notna()].copy()
    
    print(f"✓ Matched {matched_count:,} trips to routes")
    print(f"✓ Removed {len(trip_ids) - matched_count:,} unmatched trips")
    print(f"\nTrips by route:")
    print(df.groupby('route_id').size().sort_values(ascending=False))
    
    return df


# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================

def add_distance_features(df):
    """
    Calculate cumulative distance and segment distances
    
    Args:
        df: DataFrame with trip_id
    
    Returns:
        DataFrame with distance columns added
    """
    print("\n" + "="*80)
    print("STEP 4: DISTANCE FEATURES")
    print("="*80)
    
    df = df.copy()
    df['segment_distance_m'] = 0.0
    df['cum_dist_m'] = 0.0
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Calculating distances"):
        mask = df['trip_id'] == trip_id
        trip_data = df[mask].copy()
        
        # Calculate segment distances
        for i in range(1, len(trip_data)):
            prev_idx = trip_data.index[i-1]
            curr_idx = trip_data.index[i]
            
            dist = haversine_distance(
                df.loc[prev_idx, 'Latitude'], df.loc[prev_idx, 'Longitude'],
                df.loc[curr_idx, 'Latitude'], df.loc[curr_idx, 'Longitude']
            )
            df.loc[curr_idx, 'segment_distance_m'] = dist
        
        # Calculate cumulative distance
        df.loc[mask, 'cum_dist_m'] = df.loc[mask, 'segment_distance_m'].cumsum()
    
    print(f"✓ Added distance features")
    print(f"  Average trip distance: {df.groupby('trip_id')['cum_dist_m'].max().mean():.0f} meters")
    
    return df


def add_temporal_features(df):
    """
    Add time-based features
    
    Args:
        df: DataFrame with DateTime column
    
    Returns:
        DataFrame with temporal features added
    """
    print("\n" + "="*80)
    print("STEP 5: TEMPORAL FEATURES")
    print("="*80)
    
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = df['hour'].isin([8, 9, 17, 18]).astype(int)
    df['minutes_since_midnight'] = df['hour'] * 60 + df['DateTime'].dt.minute
    
    print(f"✓ Added temporal features: hour, day_of_week, is_weekend, is_peak_hour, minutes_since_midnight")
    
    return df


def add_speed_dynamics(df):
    """
    Add speed-based features (moving averages, acceleration)
    
    Args:
        df: DataFrame with Speed column
    
    Returns:
        DataFrame with speed dynamic features
    """
    print("\n" + "="*80)
    print("STEP 6: SPEED DYNAMICS")
    print("="*80)
    
    df = df.copy()
    df['speed_ma_1min'] = 0.0
    df['speed_ma_2min'] = 0.0
    df['acceleration'] = 0.0
    df['is_stopped'] = (df['Speed'] < MIN_SPEED_MOVING).astype(int)
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Calculating speed dynamics"):
        mask = df['trip_id'] == trip_id
        trip_data = df[mask].copy()
        
        # Moving averages (60 and 120 second windows)
        df.loc[mask, 'speed_ma_1min'] = trip_data['Speed'].rolling(window=60, min_periods=1).mean().values
        df.loc[mask, 'speed_ma_2min'] = trip_data['Speed'].rolling(window=120, min_periods=1).mean().values
        
        # Acceleration (speed change)
        df.loc[mask, 'acceleration'] = trip_data['Speed'].diff().fillna(0).values
    
    print(f"✓ Added speed dynamics: speed_ma_1min, speed_ma_2min, acceleration, is_stopped")
    
    return df


# ============================================================================
# STEP 5: TRAINING DATA GENERATION
# ============================================================================

def find_stop_arrivals(df):
    """
    Find when each trip arrived at each stop
    
    Args:
        df: DataFrame with trip_id and route_id
    
    Returns:
        DataFrame with arrival information
    """
    print("\n" + "="*80)
    print("STEP 7: FINDING STOP ARRIVALS")
    print("="*80)
    
    arrivals = []
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Finding arrivals"):
        trip_data = df[df['trip_id'] == trip_id].copy()
        route_id = trip_data['route_id'].iloc[0]
        route_stops = ROUTES[route_id]['stops']
        
        for stop_id in route_stops:
            stop_info = BUS_STOPS[stop_id]
            
            # Calculate distance to this stop for all points
            trip_data['dist_to_stop'] = trip_data.apply(
                lambda row: haversine_distance(
                    row['Latitude'], row['Longitude'],
                    stop_info['lat'], stop_info['lon']
                ), axis=1
            )
            
            # Find closest approach
            min_dist_idx = trip_data['dist_to_stop'].idxmin()
            min_dist = trip_data.loc[min_dist_idx, 'dist_to_stop']
            
            # Only count if within 100m
            if min_dist < 100:
                arrivals.append({
                    'trip_id': trip_id,
                    'stop_id': stop_id,
                    'arrival_time': trip_data.loc[min_dist_idx, 'DateTime'],
                    'arrival_cum_dist': trip_data.loc[min_dist_idx, 'cum_dist_m'],
                    'distance_to_stop_m': min_dist
                })
    
    arrivals_df = pd.DataFrame(arrivals)
    print(f"✓ Found {len(arrivals_df):,} stop arrivals")
    print(f"  Average arrivals per trip: {len(arrivals_df) / df['trip_id'].nunique():.1f}")
    
    return arrivals_df


def generate_training_data(df, arrivals_df):
    """
    Generate training examples by sampling GPS points before each arrival
    
    Args:
        df: Preprocessed GPS DataFrame
        arrivals_df: Stop arrivals DataFrame
    
    Returns:
        Training DataFrame with features and labels
    """
    print("\n" + "="*80)
    print("STEP 8: GENERATING TRAINING DATA")
    print("="*80)
    
    training_rows = []
    
    for _, arrival in tqdm(arrivals_df.iterrows(), total=len(arrivals_df), desc="Generating training rows"):
        trip_id = arrival['trip_id']
        stop_id = arrival['stop_id']
        arrival_time = arrival['arrival_time']
        
        # Get trip data before arrival
        trip_data = df[
            (df['trip_id'] == trip_id) &
            (df['DateTime'] < arrival_time) &
            (df['DateTime'] >= arrival_time - timedelta(minutes=MAX_ETA_MINUTES))
        ].copy()
        
        if len(trip_data) == 0:
            continue
        
        # Sample every N seconds
        trip_data['seconds_from_start'] = (trip_data['DateTime'] - trip_data['DateTime'].min()).dt.total_seconds()
        trip_data['sample_group'] = (trip_data['seconds_from_start'] // SAMPLING_INTERVAL_SEC).astype(int)
        sampled = trip_data.groupby('sample_group').first().reset_index(drop=True)
        
        # Calculate labels and features
        for _, row in sampled.iterrows():
            eta_seconds = (arrival_time - row['DateTime']).total_seconds()
            
            if eta_seconds <= 0:
                continue
            
            # Calculate distance to target stop
            stop_info = BUS_STOPS[stop_id]
            dist_to_stop = haversine_distance(
                row['Latitude'], row['Longitude'],
                stop_info['lat'], stop_info['lon']
            )
            
            # Get route info
            route_id = row['route_id']
            route_stops = ROUTES[route_id]['stops']
            stop_sequence = route_stops.index(stop_id) if stop_id in route_stops else -1
            
            training_rows.append({
                # Identifiers
                'trip_id': trip_id,
                'stop_id': stop_id,
                'route_id': route_id,
                
                # Current state
                'latitude': row['Latitude'],
                'longitude': row['Longitude'],
                'speed': row['Speed'],
                'acceleration': row['acceleration'],
                
                # Distance features
                'cum_dist_m': row['cum_dist_m'],
                'dist_to_stop_m': dist_to_stop,
                
                # Temporal features
                'hour': row['hour'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'is_peak_hour': row['is_peak_hour'],
                'minutes_since_midnight': row['minutes_since_midnight'],
                
                # Speed dynamics
                'speed_ma_1min': row['speed_ma_1min'],
                'speed_ma_2min': row['speed_ma_2min'],
                'is_stopped': row['is_stopped'],
                
                # Route context
                'stop_sequence': stop_sequence,
                
                # LABEL
                'ETA_sec': eta_seconds
            })
    
    training_df = pd.DataFrame(training_rows)
    
    print(f"✓ Generated {len(training_df):,} training examples")
    print(f"  Average ETA: {training_df['ETA_sec'].mean()/60:.1f} minutes")
    print(f"  Median ETA: {training_df['ETA_sec'].median()/60:.1f} minutes")
    
    return training_df


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_data(input_df, output_path='processed_training_data.csv'):
    """
    Complete preprocessing pipeline
    
    Args:
        input_df: Raw GPS DataFrame from data_loader
        output_path: Path to save processed CSV
    
    Returns:
        Processed training DataFrame
    """
    print("\n" + "="*80)
    print("BUS ETA PREDICTION - DATA PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Clean data
    df = clean_gps_data(input_df)
    
    # Step 2: Detect trips
    df = detect_trips(df)
    
    # Step 3: Match routes
    df = assign_routes(df)
    
    # Step 4-6: Add features
    df = add_distance_features(df)
    df = add_temporal_features(df)
    df = add_speed_dynamics(df)
    
    # Step 7: Find arrivals
    arrivals_df = find_stop_arrivals(df)
    
    # Step 8: Generate training data
    training_df = generate_training_data(df, arrivals_df)
    
    # Save to CSV
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)
    
    training_df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Summary statistics
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Total training examples: {len(training_df):,}")
    print(f"Unique trips: {training_df['trip_id'].nunique():,}")
    print(f"Unique stops: {training_df['stop_id'].nunique()}")
    print(f"Routes covered: {training_df['route_id'].nunique()}")
    print(f"\nFeatures: {len(training_df.columns) - 1}")  # Exclude ETA_sec
    print(f"Label: ETA_sec (range: {training_df['ETA_sec'].min():.0f} - {training_df['ETA_sec'].max():.0f} seconds)")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return training_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    from data_loader import load_single_day
    
    print("Loading GPS data...")
    df = load_single_day('20251103')  # Load one day for testing
    
    print(f"Loaded {len(df):,} GPS records")
    
    # Run preprocessing
    training_data = preprocess_data(df, output_path='training_data.csv')
    
    print("\n✓ Preprocessing complete!")
    print(f"Training data ready: training_data.csv")
