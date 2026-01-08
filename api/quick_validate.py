"""
Quick Validation Test - Per-IMEI Processing with Timing
Tests runtime and accuracy by processing each IMEI file separately
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Import production modules
from predictor import EnsembleModel
from feature_extractor import LiveDataPreprocessor

print("="*80)
print("QUICK VALIDATION TEST - Per-IMEI Runtime Analysis")
print("="*80)
print(f"\n📅 Test Date: 2025-11-24")
print(f"⏱️  Processing: Each IMEI separately with 1-minute filter\n")

# 1. Load model once (reused for all IMEIs)
print("🔄 Loading ensemble model...")
model_start = time.time()
model = EnsembleModel()
preprocessor = LiveDataPreprocessor()
model_load_time = time.time() - model_start
print(f"  ⏱️  Model load time: {model_load_time:.2f}s\n")

# 2. Find all IMEI CSV files
print("📂 Scanning for IMEI files...")
csv_files = glob.glob("../20to24nov/20251124/*.csv")

if not csv_files:
    print("❌ No CSV files found in ../20to24nov/20251124/")
    exit(1)

print(f"  Found {len(csv_files)} IMEI files\n")

# 3. Process each IMEI file separately
all_results = []
timing_summary = []

for file_idx, csv_file in enumerate(csv_files, 1):
    imei_name = Path(csv_file).stem
    print(f"{'='*80}")
    print(f"Processing IMEI {file_idx}/{len(csv_files)}: {imei_name}")
    print(f"{'='*80}")
    
    # Start timer for this IMEI
    imei_start = time.time()
    
    # Load data
    load_start = time.time()
    df = pd.read_csv(csv_file)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    load_time = time.time() - load_start
    
    print(f"  📂 Loaded: {len(df):,} GPS points ({load_time:.2f}s)")
    
    if len(df) == 0:
        print(f"  ⚠️  Skipping - no data\n")
        continue
    
    # Filter to 1 minute of data
    filter_start = time.time()
    start_dt = df['DateTime'].min()
    end_dt = start_dt + timedelta(minutes=1)
    df_1min = df[(df['DateTime'] >= start_dt) & (df['DateTime'] < end_dt)]
    filter_time = time.time() - filter_start
    
    print(f"  ⏱️  Filtered to 1 minute: {len(df_1min)} points ({filter_time:.3f}s)")
    print(f"     Time range: {start_dt.strftime('%H:%M:%S')} to {end_dt.strftime('%H:%M:%S')}")
    
    if len(df_1min) == 0:
        print(f"  ⚠️  Skipping - no data in 1-minute window\n")
        continue
    
    # Generate predictions
    predict_start = time.time()
    predictions = []
    errors = []
    
    # Debug: Print first GPS point
    if len(df_1min) > 0:
        first_row = df_1min.iloc[0]
        print(f"     Sample GPS: lat={first_row['Latitude']:.6f}, lon={first_row['Longitude']:.6f}, speed={first_row['Speed']}")
    
    # Track validation failures and fallbacks
    validation_failures = 0
    fallbacks = 0
    
    for idx, row in df_1min.iterrows():
        gps_data = {
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'speed': row['Speed'],
            'timestamp': row['DateTime']
        }
        
        try:
            # Try to create graph (includes validation)
            graph = preprocessor.gps_to_graph(gps_data, target_stop_id=18)
            
            # Use production predict_with_bounds
            result = model.predict_with_bounds(graph)
            
            predictions.append({
                'imei': imei_name,
                'timestamp': row['DateTime'],
                'latitude': row['Latitude'],
                'longitude': row['Longitude'],
                'speed': row['Speed'],
                'predicted_eta_seconds': result['eta_seconds'],
                'predicted_eta_minutes': result['eta_minutes'],
                'confidence': result['confidence'],
                'method': result['method'],
                'was_capped': result.get('was_capped', False)
            })
            
            if result['method'] != 'ml_model':
                fallbacks += 1
                
        except ValueError as ve:
            # GPS validation failed - use fallback
            validation_failures += 1
            from prediction_utils import simple_eta_fallback
            
            try:
                result = simple_eta_fallback(
                    row['Latitude'],
                    row['Longitude'],
                    18  # Main Gate
                )
                
                predictions.append({
                    'imei': imei_name,
                    'timestamp': row['DateTime'],
                    'latitude': row['Latitude'],
                    'longitude': row['Longitude'],
                    'speed': row['Speed'],
                    'predicted_eta_seconds': result['eta_seconds'],
                    'predicted_eta_minutes': result['eta_minutes'],
                    'confidence': result['confidence'],
                    'method': 'validation_failed_fallback',
                    'was_capped': result.get('was_capped', False)
                })
                fallbacks += 1
            except:
                errors.append(f"Validation failed: {str(ve)}")
                continue
                
        except Exception as e:
            errors.append(str(e))
            continue
    
    predict_time = time.time() - predict_start
    total_imei_time = time.time() - imei_start
    
    # Calculate metrics
    if predictions:
        pred_df = pd.DataFrame(predictions)
        mean_eta = pred_df['predicted_eta_minutes'].mean()
        time_per_point = (predict_time / len(predictions)) * 1000
        
        print(f"  🔮 Predictions: {len(predictions)} generated ({predict_time:.2f}s)")
        print(f"     Time per point: {time_per_point:.1f}ms")
        print(f"     Mean ETA: {mean_eta:.2f} minutes")
        
        if validation_failures > 0:
            print(f"     ⚠️  Validation failures: {validation_failures} (used fallback)")
        if fallbacks > 0:
            print(f"     🔄 Fallbacks used: {fallbacks}")
        
        all_results.extend(predictions)
        
        # Store timing info
        timing_summary.append({
            'imei': imei_name,
            'points': len(df_1min),
            'predictions': len(predictions),
            'load_time': load_time,
            'filter_time': filter_time,
            'predict_time': predict_time,
            'total_time': total_imei_time,
            'ms_per_point': time_per_point,
            'mean_eta': mean_eta
        })
    else:
        print(f"  ⚠️  No predictions generated")
        if errors:
            # Show unique errors
            unique_errors = list(set(errors))[:3]
            print(f"     Errors ({len(errors)} total):")
            for err in unique_errors:
                print(f"       - {err[:80]}")
    
    print(f"  ⏱️  Total IMEI processing time: {total_imei_time:.2f}s\n")

# 4. Overall Summary
print("="*80)
print("OVERALL RESULTS")
print("="*80)

if all_results:
    results_df = pd.DataFrame(all_results)
    timing_df = pd.DataFrame(timing_summary)
    
    # Timing statistics
    print(f"\n⏱️  Performance Summary:")
    print(f"  IMEIs processed: {len(timing_df)}")
    print(f"  Total points: {timing_df['points'].sum():,}")
    print(f"  Total predictions: {timing_df['predictions'].sum():,}")
    print(f"  Total time: {timing_df['total_time'].sum():.2f}s")
    print(f"  Avg time per IMEI: {timing_df['total_time'].mean():.2f}s")
    print(f"  Avg time per point: {timing_df['ms_per_point'].mean():.1f}ms")
    
    # Production metrics
    print(f"\n🛡️  Production Metrics:")
    total_preds = len(results_df)
    
    # Count by method
    if 'method' in results_df.columns:
        ml_count = (results_df['method'] == 'ml_model').sum()
        fallback_count = total_preds - ml_count
        print(f"  ML predictions: {ml_count} ({ml_count/total_preds*100:.1f}%)")
        print(f"  Fallback predictions: {fallback_count} ({fallback_count/total_preds*100:.1f}%)")
    
    # Count capped predictions
    if 'was_capped' in results_df.columns:
        capped_count = results_df['was_capped'].sum()
        print(f"  Capped predictions: {capped_count} ({capped_count/total_preds*100:.1f}%)")
    
    # Confidence stats
    if 'confidence' in results_df.columns:
        print(f"  Avg confidence: {results_df['confidence'].mean():.2f}")
        low_conf = (results_df['confidence'] < 0.6).sum()
        print(f"  Low confidence (<0.6): {low_conf} ({low_conf/total_preds*100:.1f}%)")
    
    # Prediction statistics
    print(f"\n📊 Prediction Statistics (All IMEIs):")
    print(f"  Mean ETA: {results_df['predicted_eta_minutes'].mean():.2f} minutes")
    print(f"  Median ETA: {results_df['predicted_eta_minutes'].median():.2f} minutes")
    print(f"  Std Dev: {results_df['predicted_eta_minutes'].std():.2f} minutes")
    print(f"  Max ETA: {results_df['predicted_eta_minutes'].max():.2f} minutes")
    print(f"  Min ETA: {results_df['predicted_eta_minutes'].min():.2f} minutes")
    
    # Per-IMEI breakdown
    print(f"\n📋 Per-IMEI Breakdown:")
    print(f"{'IMEI':<30} {'Points':<8} {'Predict Time':<15} {'ms/point':<10}")
    print("-" * 70)
    for _, row in timing_df.iterrows():
        imei_short = row['imei'][:30]
        print(f"{imei_short:<30} {row['predictions']:<8} {row['predict_time']:<15.2f} {row['ms_per_point']:<10.1f}")
    
    # Save results
    results_df.to_csv('quick_validation_results.csv', index=False)
    timing_df.to_csv('quick_validation_timing.csv', index=False)
    print(f"\n💾 Saved results:")
    print(f"   - quick_validation_results.csv (predictions)")
    print(f"   - quick_validation_timing.csv (timing breakdown)")
    
    # Extrapolation
    avg_ms_per_point = timing_df['ms_per_point'].mean()
    print(f"\n🚀 Real-Time Extrapolation:")
    print(f"  10 buses × latest point: {avg_ms_per_point * 10:.0f}ms")
    print(f"  10 buses × 60 points (sequential): {avg_ms_per_point * 600 / 1000:.1f}s")
    print(f"  10 buses × 60 points (parallel 10x): {avg_ms_per_point * 60 / 1000:.1f}s")
    
    # CREATE PERFORMANCE VISUALIZATIONS
    print(f"\n📊 Generating performance visualizations...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Processing Time per IMEI
    ax1 = plt.subplot(2, 3, 1)
    top_20 = timing_df.nlargest(20, 'total_time')
    plt.barh(range(len(top_20)), top_20['total_time'], color='steelblue', edgecolor='black')
    plt.yticks(range(len(top_20)), [name[:25] for name in top_20['imei']], fontsize=8)
    plt.xlabel('Total Time (seconds)', fontsize=10, fontweight='bold')
    plt.title('Top 20 Slowest IMEIs', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 2. ms per Point Distribution
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(timing_df['ms_per_point'], bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.axvline(x=timing_df['ms_per_point'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {timing_df["ms_per_point"].mean():.1f}ms')
    plt.xlabel('Milliseconds per Point', fontsize=10, fontweight='bold')
    plt.ylabel('Frequency', fontsize=10, fontweight='bold')
    plt.title('Prediction Speed Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 3. ETA Distribution
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(results_df['predicted_eta_minutes'], bins=50, edgecolor='black', 
             alpha=0.7, color='lightgreen')
    plt.axvline(x=results_df['predicted_eta_minutes'].median(), color='red', 
                linestyle='--', linewidth=2, 
                label=f'Median: {results_df["predicted_eta_minutes"].median():.1f} min')
    plt.xlabel('Predicted ETA (minutes)', fontsize=10, fontweight='bold')
    plt.ylabel('Frequency', fontsize=10, fontweight='bold')
    plt.title('ETA Prediction Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. Points vs Processing Time
    ax4 = plt.subplot(2, 3, 4)
    plt.scatter(timing_df['predictions'], timing_df['predict_time'], 
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.xlabel('Number of Points', fontsize=10, fontweight='bold')
    plt.ylabel('Processing Time (seconds)', fontsize=10, fontweight='bold')
    plt.title('Points vs Processing Time', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. Performance Metrics Summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    metrics_text = f"""
PERFORMANCE METRICS

Total IMEIs: {len(timing_df)}
Total Predictions: {timing_df['predictions'].sum():,}

Speed Metrics:
  Avg ms/point: {timing_df['ms_per_point'].mean():.1f}ms
  Min ms/point: {timing_df['ms_per_point'].min():.1f}ms
  Max ms/point: {timing_df['ms_per_point'].max():.1f}ms

Real-Time Capacity:
  10 buses (latest): {avg_ms_per_point * 10:.0f}ms
  10 buses (60pts):  {avg_ms_per_point * 600 / 1000:.1f}s

ETA Statistics:
  Mean:   {results_df['predicted_eta_minutes'].mean():.1f} min
  Median: {results_df['predicted_eta_minutes'].median():.1f} min
  Std:    {results_df['predicted_eta_minutes'].std():.1f} min
"""
    plt.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6. Cumulative Processing Time
    ax6 = plt.subplot(2, 3, 6)
    timing_sorted = timing_df.sort_values('total_time', ascending=False).reset_index(drop=True)
    cumulative_time = timing_sorted['total_time'].cumsum()
    plt.plot(range(len(cumulative_time)), cumulative_time, linewidth=2, color='darkblue')
    plt.xlabel('Number of IMEIs Processed', fontsize=10, fontweight='bold')
    plt.ylabel('Cumulative Time (seconds)', fontsize=10, fontweight='bold')
    plt.title('Cumulative Processing Time', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Quick Validation Performance Analysis', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save visualization
    viz_path = 'quick_validation_performance.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"   - {viz_path} (performance visualization)")
    plt.close()
    
    print("\n" + "="*80)
    print("✅ Quick validation complete with visualizations!")
    print("="*80)
    print(f"\n📊 Open '{viz_path}' to see performance charts!")
else:
    print("❌ No predictions generated")

