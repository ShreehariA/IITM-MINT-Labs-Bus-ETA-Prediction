"""
Test script for preprocessing pipeline
Run this in your Jupyter notebook (test.ipynb) where pandas is already installed

This will:
1. Load November 3, 2025 data
2. Run the preprocessing pipeline
3. Generate test_training_data.csv
4. Validate the output
"""

from data_loader import load_single_day
from preprocessing import preprocess_data
import pandas as pd

print("="*80)
print("PREPROCESSING PIPELINE TEST")
print("="*80)

# Step 1: Load data
print("\n1. Loading GPS data for November 3, 2025...")
df = load_single_day('20251103')
print(f"   ✓ Loaded {len(df):,} GPS records")

# Step 2: Run preprocessing
print("\n2. Running preprocessing pipeline...")
training_data = preprocess_data(df, output_path='test_training_data.csv')

# Step 3: Validation
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

# Check 1: Data generated
assert len(training_data) > 0, "❌ No training data generated"
print(f"✓ Generated {len(training_data):,} training examples")

# Check 2: Required columns
required_cols = ['ETA_sec', 'latitude', 'longitude', 'speed', 'route_id', 'stop_id']
for col in required_cols:
    assert col in training_data.columns, f"❌ Missing column: {col}"
print(f"✓ All required columns present")

# Check 3: No missing values
missing = training_data.isnull().sum().sum()
assert missing == 0, f"❌ Found {missing} missing values"
print(f"✓ No missing values")

# Check 4: ETA range
assert training_data['ETA_sec'].min() >= 0, "❌ Negative ETA values"
assert training_data['ETA_sec'].max() <= 1800, "❌ ETA > 30 minutes"
print(f"✓ ETA values in valid range (0-1800 seconds)")

# Check 5: GPS bounds
assert training_data['latitude'].between(12.98, 13.06).all(), "❌ Invalid latitudes"
assert training_data['longitude'].between(80.22, 80.29).all(), "❌ Invalid longitudes"
print(f"✓ GPS coordinates within IITM bounds")

# Check 6: Speed range
assert training_data['speed'].between(0, 60).all(), "❌ Invalid speeds"
print(f"✓ Speed values in valid range (0-60 km/h)")

# Summary statistics
print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)
print(f"Total training examples: {len(training_data):,}")
print(f"Unique trips: {training_data['trip_id'].nunique():,}")
print(f"Unique stops: {training_data['stop_id'].nunique()}")
print(f"Routes covered: {sorted(training_data['route_id'].unique())}")
print(f"\nFeatures: {len(training_data.columns) - 1}")
print(f"Label: ETA_sec")

print(f"\nETA Statistics (minutes):")
print(training_data['ETA_sec'].describe() / 60)

print(f"\nSpeed Statistics (km/h):")
print(training_data['speed'].describe())

print("\n" + "="*80)
print("✓ ALL TESTS PASSED!")
print("="*80)
print(f"\nOutput saved to: test_training_data.csv")

print("\n✓ Ready to process full dataset!")
