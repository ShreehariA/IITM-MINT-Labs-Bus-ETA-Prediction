"""
Test the date-based train/test split improvement

This script tests Priority 1 of Quick Wins:
- Date-based split instead of random trips
- Expected: Reduce overfitting from 28.5% to ~15%
- Expected: Improve generalization
"""

import pandas as pd
from test_xgboost import train_and_evaluate_xgboost

print("="*80)
print("TESTING PRIORITY 1: DATE-BASED TRAIN/TEST SPLIT")
print("="*80)

# Load data
df = pd.read_csv('training_full_18days_parallel.csv')

print(f"\nDataset: {len(df):,} examples")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique dates: {df['date'].nunique()}")

# Train with new split
print("\n" + "="*80)
print("TRAINING WITH DATE-BASED SPLIT")
print("="*80)

model, metrics, encoders = train_and_evaluate_xgboost(
    df,
    save_model_path='xgboost_priority1.json',
    save_plot_path='priority1_results.png'
)

print("\n" + "="*80)
print("COMPARISON: BEFORE vs AFTER")
print("="*80)

print("\nBEFORE (Random Split):")
print("  MAE Train: 4.30 min")
print("  MAE Test:  5.53 min")
print("  Overfitting: 28.5%")
print("  R² Test: 0.368")
print("  Accuracy ±5min: 52.8%")

print("\nAFTER (Date-Based Split):")
print(f"  MAE Train: {metrics['train_mae']:.2f} min")
print(f"  MAE Test:  {metrics['test_mae']:.2f} min")
print(f"  Overfitting: {abs(metrics['train_mae'] - metrics['test_mae']) / metrics['test_mae'] * 100:.1f}%")
print(f"  R² Test: {metrics['test_r2']:.3f}")
print(f"  Accuracy ±5min: {metrics.get('accuracy_5min', 'N/A')}%")

print("\n" + "="*80)
print("✓ PRIORITY 1 COMPLETE")
print("="*80)
