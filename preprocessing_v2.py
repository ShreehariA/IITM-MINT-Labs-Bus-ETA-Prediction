"""
Enhanced Preprocessing with Feature Engineering v2

This version adds research-backed features while keeping the same data schema:
- Cyclical time encoding (sin/cos for hour, day)
- Lag features (previous speeds, distances)
- Rolling statistics (moving averages)
- Traffic indicators (speed variance, congestion)
- Historical patterns (same route, same time)

All features are derived from existing columns:
- IMEI, DateTime, Latitude, Longitude, Speed

No external data needed!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import from original preprocessing
from preprocessing import (
    IITM_BOUNDS, MAX_SPEED_KMH, MIN_SPEED_MOVING,
    TRIP_GAP_MINUTES, MIN_TRIP_POINTS,
    SAMPLING_INTERVAL_SEC, MAX_ETA_MINUTES,
    BUS_STOPS, ROUTES,
    haversine_distance, is_within_bounds, find_nearest_stop,
    clean_gps_data, detect_trips, assign_routes,
    find_stop_arrivals
)

# ============================================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================================

def add_cyclical_time_features(df):
    """
    Add cyclical encoding for time features
    
    Research shows: Cyclical encoding (sin/cos) performs better than linear
    for time features because it captures the circular nature of time.
    
    Example: Hour 23 and Hour 0 are close, but linearly they're far apart.
    """
    print("\n" + "="*80)
    print("ENHANCED TEMPORAL FEATURES (Cyclical Encoding)")
    print("="*80)
    
    df = df.copy()
    
    # Hour (0-23) -> cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (0-6) -> cyclical
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Keep original for compatibility
    print(f"✓ Added cyclical time features: hour_sin, hour_cos, day_sin, day_cos")
    
    return df


def add_lag_features(df):
    """
    Add lag features (previous values) for each trip
    
    Research shows: Historical context significantly improves predictions
    Lag features capture momentum and trends in bus movement.
    """
    print("\n" + "="*80)
    print("LAG FEATURES (Historical Context)")
    print("="*80)
    
    df = df.copy()
    
    # Initialize lag columns
    df['speed_lag_1'] = 0.0
    df['speed_lag_5'] = 0.0
    df['speed_lag_10'] = 0.0
    df['dist_lag_1'] = 0.0
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Adding lag features"):
        mask = df['trip_id'] == trip_id
        trip_data = df[mask].copy()
        
        # Speed lags (1, 5, 10 points back)
        df.loc[mask, 'speed_lag_1'] = trip_data['Speed'].shift(1).fillna(0)
        df.loc[mask, 'speed_lag_5'] = trip_data['Speed'].shift(5).fillna(0)
        df.loc[mask, 'speed_lag_10'] = trip_data['Speed'].shift(10).fillna(0)
        
        # Distance lag
        df.loc[mask, 'dist_lag_1'] = trip_data['segment_distance_m'].shift(1).fillna(0)
    
    print(f"✓ Added lag features: speed_lag_1, speed_lag_5, speed_lag_10, dist_lag_1")
    
    return df


def add_traffic_indicators(df):
    """
    Add traffic/congestion indicators
    
    Research shows: Traffic dynamics are critical for ETA prediction
    Speed variance and time in congestion are strong predictors.
    """
    print("\n" + "="*80)
    print("TRAFFIC INDICATORS")
    print("="*80)
    
    df = df.copy()
    
    # Initialize columns
    df['speed_variance_5min'] = 0.0
    df['time_below_5kmh_pct'] = 0.0
    df['num_speed_drops'] = 0
    df['congestion_score'] = 0.0
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Calculating traffic indicators"):
        mask = df['trip_id'] == trip_id
        trip_data = df[mask].copy()
        
        # Speed variance (last 5 minutes = 300 points)
        df.loc[mask, 'speed_variance_5min'] = trip_data['Speed'].rolling(
            window=300, min_periods=1
        ).var().fillna(0)
        
        # Time spent below 5 km/h (congestion indicator)
        df.loc[mask, 'time_below_5kmh_pct'] = trip_data['Speed'].rolling(
            window=300, min_periods=1
        ).apply(lambda x: (x < 5).sum() / len(x) * 100, raw=True).fillna(0)
        
        # Number of sudden speed drops (>10 km/h drop)
        speed_diff = trip_data['Speed'].diff()
        df.loc[mask, 'num_speed_drops'] = (speed_diff < -10).rolling(
            window=60, min_periods=1
        ).sum().fillna(0)
        
        # Congestion score (composite)
        df.loc[mask, 'congestion_score'] = (
            df.loc[mask, 'time_below_5kmh_pct'] / 100 * 0.5 +
            (df.loc[mask, 'speed_variance_5min'] / 100).clip(0, 1) * 0.3 +
            (df.loc[mask, 'num_speed_drops'] / 10).clip(0, 1) * 0.2
        )
    
    print(f"✓ Added traffic indicators: speed_variance_5min, time_below_5kmh_pct, num_speed_drops, congestion_score")
    
    return df


def add_improved_stop_features(df):
    """
    Replace is_stopped with better features
    
    Research shows: is_stopped causes data leakage
    Better to use time_since_last_movement and movement_state
    """
    print("\n" + "="*80)
    print("IMPROVED STOP FEATURES (Replacing is_stopped)")
    print("="*80)
    
    df = df.copy()
    
    # Remove old is_stopped
    if 'is_stopped' in df.columns:
        df = df.drop('is_stopped', axis=1)
    
    # Add better features
    df['time_since_last_movement'] = 0
    df['num_stops_last_5min'] = 0
    
    for trip_id in tqdm(df['trip_id'].unique(), desc="Calculating stop features"):
        mask = df['trip_id'] == trip_id
        trip_data = df[mask].copy()
        
        # Time since last movement
        is_moving = trip_data['Speed'] >= MIN_SPEED_MOVING
        time_counter = 0
        times = []
        for moving in is_moving:
            if moving:
                time_counter = 0
            else:
                time_counter += 1
            times.append(time_counter)
        df.loc[mask, 'time_since_last_movement'] = times
        
        # Number of stops in last 5 minutes
        df.loc[mask, 'num_stops_last_5min'] = (trip_data['Speed'] < MIN_SPEED_MOVING).rolling(
            window=300, min_periods=1
        ).sum()
    
    print(f"✓ Replaced is_stopped with: time_since_last_movement, num_stops_last_5min")
    
    return df


def add_historical_features(df):
    """
    Add historical average features
    
    Research shows: Same route, same time patterns are highly predictive
    Uses data from previous days/weeks for the same conditions.
    """
    print("\n" + "="*80)
    print("HISTORICAL PATTERN FEATURES")
    print("="*80)
    
    df = df.copy()
    
    # Group by route, hour to get historical averages
    print("  Calculating historical averages by route-hour...")
    
    # Average speed for this route at this hour
    route_hour_speed = df.groupby(['route_id', 'hour'])['Speed'].transform('mean')
    df['route_hour_avg_speed'] = route_hour_speed
    
    # Average speed for this route on weekends vs weekdays
    route_weekend_speed = df.groupby(['route_id', 'is_weekend'])['Speed'].transform('mean')
    df['route_weekend_avg_speed'] = route_weekend_speed
    
    # Average cumulative distance for this route
    route_avg_dist = df.groupby('route_id')['cum_dist_m'].transform('mean')
    df['route_avg_cum_dist'] = route_avg_dist
    
    print(f"✓ Added historical features: route_hour_avg_speed, route_weekend_avg_speed, route_avg_cum_dist")
    
    return df


# ============================================================================
# ENHANCED TRAINING DATA GENERATION
# ============================================================================

def generate_enhanced_training_data(df, arrivals_df):
    """
    Generate training data with all enhanced features
    
    Same as original but includes all new features
    """
    print("\n" + "="*80)
    print("GENERATING ENHANCED TRAINING DATA")
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
                
                # Original temporal features
                'hour': row['hour'],
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'is_peak_hour': row['is_peak_hour'],
                'minutes_since_midnight': row['minutes_since_midnight'],
                
                # NEW: Cyclical time features
                'hour_sin': row['hour_sin'],
                'hour_cos': row['hour_cos'],
                'day_sin': row['day_sin'],
                'day_cos': row['day_cos'],
                
                # Speed dynamics (original)
                'speed_ma_1min': row['speed_ma_1min'],
                'speed_ma_2min': row['speed_ma_2min'],
                
                # NEW: Lag features
                'speed_lag_1': row['speed_lag_1'],
                'speed_lag_5': row['speed_lag_5'],
                'speed_lag_10': row['speed_lag_10'],
                'dist_lag_1': row['dist_lag_1'],
                
                # NEW: Traffic indicators
                'speed_variance_5min': row['speed_variance_5min'],
                'time_below_5kmh_pct': row['time_below_5kmh_pct'],
                'num_speed_drops': row['num_speed_drops'],
                'congestion_score': row['congestion_score'],
                
                # NEW: Improved stop features
                'time_since_last_movement': row['time_since_last_movement'],
                'num_stops_last_5min': row['num_stops_last_5min'],
                
                # NEW: Historical features
                'route_hour_avg_speed': row['route_hour_avg_speed'],
                'route_weekend_avg_speed': row['route_weekend_avg_speed'],
                'route_avg_cum_dist': row['route_avg_cum_dist'],
                
                # Route context
                'stop_sequence': stop_sequence,
                
                # LABEL
                'ETA_sec': eta_seconds
            })
    
    training_df = pd.DataFrame(training_rows)
    
    print(f"✓ Generated {len(training_df):,} training examples")
    print(f"  Features: {len(training_df.columns) - 1}")  # Exclude ETA_sec
    print(f"  Average ETA: {training_df['ETA_sec'].mean()/60:.1f} minutes")
    print(f"  Median ETA: {training_df['ETA_sec'].median()/60:.1f} minutes")
    
    return training_df


# ============================================================================
# MAIN ENHANCED PREPROCESSING PIPELINE
# ============================================================================

def preprocess_data_v2(input_df, output_path='training_data_enhanced.csv'):
    """
    Enhanced preprocessing pipeline with all improvements
    
    Args:
        input_df: Raw GPS DataFrame from data_loader
        output_path: Path to save processed CSV
    
    Returns:
        Enhanced training DataFrame
    """
    print("\n" + "="*80)
    print("BUS ETA PREDICTION - ENHANCED PREPROCESSING PIPELINE V2")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Original steps 1-6
    df = clean_gps_data(input_df)
    df = detect_trips(df)
    df = assign_routes(df)
    
    # Add distance features (from original)
    from preprocessing import add_distance_features, add_temporal_features, add_speed_dynamics
    df = add_distance_features(df)
    df = add_temporal_features(df)
    df = add_speed_dynamics(df)
    
    # NEW: Enhanced features
    df = add_cyclical_time_features(df)
    df = add_lag_features(df)
    df = add_traffic_indicators(df)
    df = add_improved_stop_features(df)
    df = add_historical_features(df)
    
    # Find arrivals
    arrivals_df = find_stop_arrivals(df)
    
    # Generate enhanced training data
    training_df = generate_enhanced_training_data(df, arrivals_df)
    
    # Save to CSV
    print("\n" + "="*80)
    print("SAVING ENHANCED DATA")
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
    print(f"\nFeatures: {len(training_df.columns) - 1}")
    print(f"  Original features: 18")
    print(f"  NEW features: {len(training_df.columns) - 19}")
    print(f"Label: ETA_sec (range: {training_df['ETA_sec'].min():.0f} - {training_df['ETA_sec'].max():.0f} seconds)")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return training_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from data_loader import load_multiple_days
    
    print("Loading 1 week of GPS data...")
    dates = ['20251103', '20251104', '20251105', '20251106', 
             '20251107', '20251108', '20251109']
    df = load_multiple_days(dates)
    
    print(f"Loaded {len(df):,} GPS records")
    
    # Run enhanced preprocessing
    training_data = preprocess_data_v2(df, output_path='training_data_enhanced_1week.csv')
    
    print("\n✓ Enhanced preprocessing complete!")
    print(f"Training data ready: training_data_enhanced_1week.csv")
