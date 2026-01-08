"""
PARALLEL FULL DATASET TRAINING PIPELINE
Uses multiprocessing to speed up data loading and preprocessing

Strategy:
1. Process days in parallel (4-6 workers)
2. Each worker: load → preprocess → append to CSV
3. Simplified route matching (no complex stop detection)
4. Final: Train XGBoost on combined CSV

Author: Shreehari Anbazhagan
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from functools import partial
import warnings
import signal
import sys
warnings.filterwarnings('ignore')

from data_loader import load_single_day
from preprocessing_unified import (
    clean_gps_data, detect_trips, add_distance_features,
    add_temporal_features, add_speed_dynamics
)

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR M4 PRO
# ============================================================================

# All 18 days
ALL_DATES = [
    '20251103', '20251104', '20251106', '20251107', '20251108', '20251109',  # Week 1
    '20251110', '20251111', '20251112', '20251113', '20251114', '20251115',  # Week 2
    '20251116', '20251117', '20251118', '20251119', '20251120', '20251121',  # Week 3
]

OUTPUT_CSV = 'training_full_18days_parallel.csv'

# M4 PRO OPTIMIZATION (14 cores, 24GB RAM)
NUM_WORKERS = 12  # Use 12 of 14 cores (leave 2 for system)
CHUNK_SIZE = 1    # Process 1 day per worker at a time for better distribution

# Simplified stop detection (just use GPS clusters)
STOP_THRESHOLD_KMH = 2.0  # Speed below this = stopped
STOP_DURATION_SEC = 30    # Stopped for this long = arrival

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_historical_speed_features(df):
    """
    Add historical speed patterns per route segment
    Priority 3: Historical speed features
    """
    # Create segment ID from lat/lon (100m grid)
    df['segment_id'] = (
        (df['latitude'] * 100).astype(int).astype(str) + '_' +
        (df['longitude'] * 100).astype(int).astype(str)
    )
    
    # Calculate historical speed per segment per hour
    historical = df.groupby(['segment_id', 'hour'])['speed'].agg([
        ('hist_speed_mean', 'mean'),
        ('hist_speed_std', 'std'),
        ('hist_speed_p25', lambda x: x.quantile(0.25)),
        ('hist_speed_p75', lambda x: x.quantile(0.75))
    ]).reset_index()
    
    # Merge back
    df = df.merge(historical, on=['segment_id', 'hour'], how='left')
    
    # Fill missing with global averages
    df['hist_speed_mean'].fillna(df['speed'].mean(), inplace=True)
    df['hist_speed_std'].fillna(df['speed'].std(), inplace=True)
    df['hist_speed_p25'].fillna(df['speed'].quantile(0.25), inplace=True)
    df['hist_speed_p75'].fillna(df['speed'].quantile(0.75), inplace=True)
    
    # Add derived features
    df['speed_vs_historical'] = df['speed'] / (df['hist_speed_mean'] + 0.1)
    df['is_slower_than_usual'] = (df['speed'] < df['hist_speed_p25']).astype(int)
    df['is_faster_than_usual'] = (df['speed'] > df['hist_speed_p75']).astype(int)
    
    return df

# DISABLED: Quick Wins made accuracy worse!
# def add_route_progress_features(df):
#     df['trip_progress_pct'] = df['cum_dist_m'] / (df.groupby('trip_id')['cum_dist_m'].transform('max') + 1)
#     df['time_into_trip'] = (df['datetime'] - df.groupby('trip_id')['datetime'].transform('min')).dt.total_seconds()
#     df['avg_speed_so_far'] = df['cum_dist_m'] / (df['time_into_trip'] + 1)
#     return df

# def add_traffic_pattern_features(df):
#     df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
#     df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
#     df['is_lunch_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 14)).astype(int)
#     max_speed_by_hour = df.groupby(['hour', 'day_of_week'])['speed'].transform('max')
#     df['congestion_factor'] = 1.0 - (df['speed'] / (max_speed_by_hour + 0.1))
#     return df

# def add_speed_trend_features(df):
#     df['speed_trend'] = df.groupby('trip_id')['speed'].diff(6).fillna(0)
#     df['is_slowing_down'] = (df['speed_trend'] < -2).astype(int)
#     df['is_speeding_up'] = (df['speed_trend'] > 2).astype(int)
#     return df

def remove_outliers(df):
    """
    Quick Win 4: Remove outliers
    Filter out clearly wrong training examples
    UPDATED: Less aggressive thresholds to preserve good data
    """
    initial_count = len(df)
    
    df = df[
        (df['ETA_sec'] > 0) &            # Negative ETA impossible
        (df['ETA_sec'] < 7200) &         # > 2 hours unrealistic (was 1 hour - too strict!)
        (df['speed'] < 80) &             # > 80 km/h is GPS error (was 60 - too strict!)
        (df['dist_to_stop_m'] > 5) &     # Very close to stop (was 10)
        (df['dist_to_stop_m'] < 10000)   # > 10km outside campus (was 5km - too strict!)
    ].copy()
    
    removed = initial_count - len(df)
    if removed > 0:
        pct = removed/initial_count*100
        print(f"  Removed {removed:,} outliers ({pct:.1f}%)")
        if pct > 15:
            print(f"  ⚠️  WARNING: Removed >{pct:.0f}% of data - thresholds may be too strict!")
    
    return df

# ============================================================================
# SIMPLIFIED PREPROCESSING (NO COMPLEX ROUTE MATCHING)
# ============================================================================

def process_single_day(date):
    """
    Process a single day's data:
    1. Load raw GPS data
    2. Preprocess (clean, detect trips, add features)
    3. Generate training examples
    """
    try:
        print(f"\n[{date}] Starting...")
        
        # 1. Load data
        df = load_single_day(date)
        if df is None or len(df) == 0:
            print(f"[{date}] ⚠ No data")
            return None
        
        print(f"[{date}] Loaded {len(df):,} records")
        
        # 2. Basic preprocessing
        df = clean_gps_data(df)
        df = detect_trips(df)
        
        if len(df) == 0:
            print(f"[{date}] ⚠ No valid trips")
            return None
        
        df = add_distance_features(df)
        df = add_temporal_features(df)
        df = add_speed_dynamics(df)
        df = add_historical_speed_features(df)   # Priority 3 (keep)
        
        # DISABLED: Quick Wins made accuracy worse!
        # df = add_route_progress_features(df)     # Quick Win 1
        # df = add_traffic_pattern_features(df)    # Quick Win 2
        # df = add_speed_trend_features(df)        # Quick Win 3
        
        # 3. VECTORIZED: Generate training data from stop events
        training_rows = []
        
        for trip_id in df['trip_id'].unique():
            trip_data = df[df['trip_id'] == trip_id].copy()
            
            # Find stop events (speed < 2 km/h for 30+ seconds)
            trip_data['is_stopped'] = trip_data['speed'] < STOP_THRESHOLD_KMH
            trip_data['stop_group'] = (trip_data['is_stopped'] != trip_data['is_stopped'].shift()).cumsum()
            
            # NEW: Priority 2 - Count total stops in this trip
            total_stops = trip_data[trip_data['is_stopped']].groupby('stop_group').ngroups
            
            # For each stop event, create training examples from approach
            stop_counter = 0
            for group_id, stop_group in trip_data[trip_data['is_stopped']].groupby('stop_group'):
                if len(stop_group) < STOP_DURATION_SEC:
                    continue
                
                # NEW: Priority 2 - Track stops remaining
                stop_counter += 1
                stops_remaining = total_stops - stop_counter
                
                # This is a valid stop arrival
                arrival_time = stop_group['datetime'].iloc[-1]
                stop_lat = stop_group['latitude'].median()
                stop_lon = stop_group['longitude'].median()
                
                # Get data from 30 minutes before arrival
                approach_data = trip_data[
                    (trip_data['datetime'] < arrival_time) &
                    (trip_data['datetime'] >= arrival_time - pd.Timedelta(minutes=30))
                ].copy()
                
                if len(approach_data) == 0:
                    continue
                
                # Sample every 10 seconds
                approach_data['seconds_from_start'] = (
                    approach_data['datetime'] - approach_data['datetime'].min()
                ).dt.total_seconds()
                approach_data['sample_group'] = (approach_data['seconds_from_start'] // 10).astype(int)
                sampled = approach_data.groupby('sample_group').first().reset_index(drop=True)
                
                # VECTORIZED: Calculate all distances at once
                from preprocessing_unified import haversine_distance
                sampled['dist_to_stop'] = sampled.apply(
                    lambda row: haversine_distance(row['latitude'], row['longitude'], stop_lat, stop_lon),
                    axis=1
                )
                
                # VECTORIZED: Calculate all ETAs at once
                sampled['eta_seconds'] = (arrival_time - sampled['datetime']).dt.total_seconds()
                
                # Filter valid ETAs
                sampled = sampled[sampled['eta_seconds'] > 0].copy()
                
                if len(sampled) == 0:
                    continue
                
                # VECTORIZED: Calculate all derived features at once
                sampled['time_to_stop_naive'] = sampled['dist_to_stop'] / (sampled['speed'] + 0.1)
                sampled['speed_efficiency'] = sampled['speed'] / (sampled['speed_ma_1min'] + 0.1)
                sampled['is_very_close'] = (sampled['dist_to_stop'] < 200).astype(int)
                sampled['is_close'] = (sampled['dist_to_stop'] < 500).astype(int)
                sampled['is_far'] = (sampled['dist_to_stop'] > 2000).astype(int)
                sampled['is_moving'] = (sampled['speed'] >= 1.0).astype(int)
                sampled['is_slow'] = ((sampled['speed'] > 0) & (sampled['speed'] < 5)).astype(int)
                sampled['is_accelerating'] = (sampled['acceleration'] > 2).astype(int)
                sampled['is_decelerating'] = (sampled['acceleration'] < -2).astype(int)
                
                # Add identifiers
                sampled['trip_id'] = trip_id
                sampled['date'] = date
                
                # Add dummy route/stop info (for compatibility with test_xgboost.py)
                sampled['route_id'] = 'AUTO_DETECTED'
                sampled['stop_id'] = int(group_id)  # Convert to int to avoid categorical issues
                sampled['stop_sequence'] = 0   # Unknown sequence
                
                # NEW: Priority 2 - Add stops remaining
                sampled['stops_remaining'] = stops_remaining
                
                # Select and rename columns
                training_batch = sampled[[
                    'trip_id', 'date', 'route_id', 'stop_id', 'stop_sequence',
                    'latitude', 'longitude', 'speed', 'acceleration',
                    'cum_dist_m', 'dist_to_stop', 'hour', 'day_of_week', 'is_weekend',
                    'is_peak_hour', 'minutes_since_midnight', 'hour_sin', 'hour_cos',
                    'day_sin', 'day_cos', 'speed_ma_30sec', 'speed_ma_1min',
                    'time_to_stop_naive', 'speed_efficiency', 'is_very_close',
                    'is_close', 'is_far', 'is_moving', 'is_slow', 'is_accelerating',
                    'is_decelerating',
                    # Priority 2 & 3 (keep these)
                    'stops_remaining', 'hist_speed_mean', 'hist_speed_std',
                    'speed_vs_historical', 'is_slower_than_usual', 'is_faster_than_usual',
                    # Quick Wins 1-3 REMOVED
                    'eta_seconds'
                ]].rename(columns={'dist_to_stop': 'dist_to_stop_m', 'eta_seconds': 'ETA_sec'})
                
                training_rows.append(training_batch)
        
        if len(training_rows) == 0:
            print(f"[{date}] ⚠ No training examples")
            return None
        
        # Concatenate all batches
        result_df = pd.concat(training_rows, ignore_index=True)
        
        # Quick Win 4: Remove outliers
        result_df = remove_outliers(result_df)
        
        print(f"[{date}] ✓ Generated {len(result_df):,} examples")
        
        return result_df
        
    except Exception as e:
        print(f"[{date}] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# PARALLEL PROCESSING PIPELINE
# ============================================================================

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print("\n\n⚠️  Shutdown requested... cleaning up workers...")
    shutdown_requested = True

def train_parallel():
    """
    Main parallel training pipeline with graceful shutdown
    """
    global shutdown_requested
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("="*80)
    print("🚀 PARALLEL FULL DATASET TRAINING (18 DAYS)")
    print("="*80)
    print(f"Workers: {NUM_WORKERS}")
    print(f"Dates: {len(ALL_DATES)}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)
    
    # Remove old output file
    if Path(OUTPUT_CSV).exists():
        Path(OUTPUT_CSV).unlink()
        print(f"✓ Removed old {OUTPUT_CSV}")
    
    # Process days in parallel with optimal chunking
    print(f"\n📊 Processing {len(ALL_DATES)} days with {NUM_WORKERS} workers...")
    print(f"💪 M4 Pro Mode: Using {NUM_WORKERS}/{mp.cpu_count()} cores")
    print("ℹ️  Press Ctrl+C to stop gracefully\n")
    
    start_time = datetime.now()
    pool = None
    results = []
    
    try:
        pool = mp.Pool(NUM_WORKERS)
        
        # Use chunksize for better distribution
        results = pool.map(process_single_day, ALL_DATES, chunksize=CHUNK_SIZE)
        
        pool.close()
        pool.join()
        
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received!")
        if pool:
            print("🛑 Terminating workers...")
            pool.terminate()
            pool.join()
        print("✓ Workers terminated cleanly")
        return None, None, None
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        if pool:
            pool.terminate()
            pool.join()
        raise
        
    finally:
        if pool:
            pool.close()
    
    if shutdown_requested:
        print("\n✓ Shutdown completed cleanly")
        return None, None, None
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱️  Processing completed in {elapsed/60:.1f} minutes")
    
    # Combine results
    print("\n" + "="*80)
    print("COMBINING RESULTS")
    print("="*80)
    
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        print("✗ No valid data generated!")
        return None, None, None
    
    print(f"✓ Got {len(valid_results)}/{len(ALL_DATES)} successful days")
    
    # Concatenate all
    training_df = pd.concat(valid_results, ignore_index=True)
    
    print(f"✓ Total training examples: {len(training_df):,}")
    print(f"  Unique trips: {training_df['trip_id'].nunique():,}")
    print(f"  Date range: {training_df['date'].min()} to {training_df['date'].max()}")
    
    # Save to CSV
    training_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved to {OUTPUT_CSV}")
    print(f"  File size: {Path(OUTPUT_CSV).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Train XGBoost
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)
    
    try:
        from test_xgboost import train_model
        
        model, metrics, encoders = train_model(
            data_path=OUTPUT_CSV,
            model_path='xgboost_model_18days.json'
        )
    except Exception as e:
        print(f"⚠️  Model training failed: {e}")
        print("✓ Training data saved successfully, you can train manually")
        return None, None, None
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"End: {datetime.now().strftime('%H:%M:%S')}")
    
    return model, metrics, encoders


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn', force=True)
    
    try:
        model, metrics, encoders = train_parallel()
        
        if model is not None:
            print("\n🎉 Model trained successfully!")
            print(f"MAE: {metrics['mae']:.2f} minutes")
            print(f"R²: {metrics['r2']:.3f}")
        else:
            print("\n⚠️  Pipeline stopped or failed")
            
    except KeyboardInterrupt:
        print("\n\n✓ Exited cleanly")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)
