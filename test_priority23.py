"""
Test Priority 2 & 3 improvements
Run this to see the accuracy improvement with new features
"""

import pandas as pd
from test_xgboost import train_and_evaluate_xgboost

print("="*80)
print("TESTING PRIORITY 2 & 3: NEW FEATURES")
print("="*80)

# Load new data
df = pd.read_csv('training_full_18days_parallel.csv')

print(f"\nDataset: {df.shape}")
print(f"Features: {df.shape[1] - 1} (was 31, added 6)")
print(f"\nNew features added:")
print("  - stops_remaining (Priority 2)")
print("  - hist_speed_mean (Priority 3)")
print("  - hist_speed_std (Priority 3)")
print("  - speed_vs_historical (Priority 3)")
print("  - is_slower_than_usual (Priority 3)")
print("  - is_faster_than_usual (Priority 3)")

# Verify new features exist
new_features = ['stops_remaining', 'hist_speed_mean', 'hist_speed_std', 
                'speed_vs_historical', 'is_slower_than_usual', 'is_faster_than_usual']
missing = [f for f in new_features if f not in df.columns]
if missing:
    print(f"\n⚠️  WARNING: Missing features: {missing}")
else:
    print("\n✓ All new features present!")

# Train model
print("\n" + "="*80)
print("TRAINING MODEL WITH NEW FEATURES")
print("="*80)

model, metrics, encoders = train_and_evaluate_xgboost(
    df,
    save_model_path='xgboost_priority23.json',
    save_plot_path='priority23_results.png'
)

# Compare results
print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

print("\nPriority 1 (Date-based split only):")
print("  Features: 31")
print("  MAE Test: 5.30 min")
print("  R² Test: 0.404")
print("  Accuracy ±5min: 55.8%")
print("  Overfitting: 23.6%")

print("\nPriority 1+2+3 (All Quick Wins):")
print(f"  Features: {df.shape[1] - 1}")
print(f"  MAE Test: {metrics['mae_test']:.2f} min")
print(f"  R² Test: {metrics['r2_test']:.3f}")
print(f"  Accuracy ±5min: {metrics['accuracy_5min']:.1f}%")
overfitting = abs(metrics['mae_train'] - metrics['mae_test']) / metrics['mae_test'] * 100
print(f"  Overfitting: {overfitting:.1f}%")

print("\nImprovement:")
print(f"  MAE: {5.30 - metrics['mae_test']:.2f} min better")
print(f"  R²: {metrics['r2_test'] - 0.404:.3f} better")
print(f"  Accuracy: {metrics['accuracy_5min'] - 55.8:.1f}% better")

print("\n" + "="*80)
print("✓ QUICK WINS COMPLETE!")
print("="*80)
