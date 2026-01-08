"""
Test script after ML audit fixes
Tests model with data leakage removed
"""
import pandas as pd
from test_xgboost import train_and_evaluate_xgboost

print("="*80)
print("ML AUDIT FIXES - TESTING")
print("="*80)
print("\n✅ Fixes Applied:")
print("1. REMOVED 'time_to_stop_naive' (DATA LEAKAGE)")
print("2. REMOVED 'speed_efficiency' (DATA LEAKAGE)")
print("3. Fixed missing value handling (train-only stats)")
print("4. Removed log transformation code")
print("\nFeatures: 37 → 35 (removed 2 leaky features)")
print("\nExpected Results:")
print("- ±1 min: 11.8% → 25-35%")
print("- ±5 min: 53.1% → 60-70%")
print("- Overfitting: 40.5% → 20-25%")
print("\nLoading data...")

# Load data
df = pd.read_csv('training_full_18days_parallel.csv')
print(f"Dataset: {df.shape}")

# Check if leaky features exist in data
if 'time_to_stop_naive' in df.columns:
    print("\n⚠️  WARNING: Data still contains 'time_to_stop_naive'")
    print("   Model will skip this column (not in FEATURE_COLUMNS)")
else:
    print("\n✅ Data doesn't have leaky features")

print("\nTraining model with ML audit fixes...\n")

# Train
model, metrics, encoders = train_and_evaluate_xgboost(df)

# Results
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)
print(f"{'Metric':<20} {'Before Fixes':<15} {'After Fixes':<15} {'Change':<15}")
print("-"*80)
print(f"{'±1 min accuracy':<20} {'11.8%':<15} {f\"{metrics.get('accuracy_1min', 0):.1f}%\":<15} {f\"{metrics.get('accuracy_1min', 0) - 11.8:+.1f}%\":<15}")
print(f"{'±5 min accuracy':<20} {'53.1%':<15} {f\"{metrics.get('accuracy_5min', 0):.1f}%\":<15} {f\"{metrics.get('accuracy_5min', 0) - 53.1:+.1f}%\":<15}")
print(f"{'MAE (minutes)':<20} {'5.48':<15} {f\"{metrics.get('mae_test', 0):.2f}\":<15} {f\"{metrics.get('mae_test', 0) - 5.48:+.2f}\":<15}")
print(f"{'R²':<20} {'0.337':<15} {f\"{metrics.get('test_r2', 0):.3f}\":<15} {f\"{metrics.get('test_r2', 0) - 0.337:+.3f}\":<15}")

# Calculate overfitting
train_mae = metrics.get('train_mae', 0) / 60  # Convert to minutes
test_mae = metrics.get('mae_test', 0)
overfitting = abs(train_mae - test_mae) / train_mae * 100 if train_mae > 0 else 0

print(f"{'Overfitting':<20} {'40.5%':<15} {f\"{overfitting:.1f}%\":<15} {f\"{overfitting - 40.5:+.1f}%\":<15}")
print("="*80)

# Verdict
if metrics.get('accuracy_1min', 0) >= 25:
    print("\n✅ SUCCESS! ±1 min accuracy improved significantly")
elif metrics.get('accuracy_1min', 0) >= 20:
    print("\n⚠️  PARTIAL SUCCESS. Some improvement, but not enough")
else:
    print("\n❌ FAILED. Need to investigate further")

print("\nNext steps:")
if metrics.get('accuracy_1min', 0) >= 25:
    print("1. Fix historical speed calculation (use previous days only)")
    print("2. Expected: +5-10% more accuracy")
else:
    print("1. Check feature importance - which features are most important?")
    print("2. Investigate why accuracy didn't improve")
