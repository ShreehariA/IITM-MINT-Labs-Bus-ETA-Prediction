"""
Batch Processing Strategy for Large Dataset (18 Days)

This script processes 18 days of data in batches to avoid memory issues,
then combines the preprocessed data for final model training.

Strategy:
1. Load and preprocess in 3 batches (6 days each)
2. Save preprocessed CSVs
3. Combine all preprocessed data
4. Train final model on combined data

Author: Shreehari Anbazhagan
Date: December 2025
"""

import pandas as pd
from data_loader import load_multiple_days
from preprocessing_unified import preprocess_data
from test_xgboost import train_and_evaluate_xgboost
import gc
from datetime import datetime

# ============================================================================
# BATCH DEFINITIONS
# ============================================================================

# Split 18 days into 3 batches of 6 days each
BATCH_1 = ['20251103', '20251104', '20251106', '20251107', '20251108', '20251109']
BATCH_2 = ['20251110', '20251111', '20251112', '20251113', '20251114', '20251115']
BATCH_3 = ['20251116', '20251117', '20251118', '20251119']  # Only 4 days (Nov 1-2 have too little data)

# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def process_batch(batch_dates, batch_name, output_file):
    """
    Load and preprocess a batch of dates
    
    Args:
        batch_dates: List of date strings
        batch_name: Name for logging
        output_file: CSV filename to save
    
    Returns:
        Number of training examples generated
    """
    print("\n" + "="*80)
    print(f"PROCESSING {batch_name}")
    print("="*80)
    print(f"Dates: {batch_dates}")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Load data
    print(f"\n[1/3] Loading GPS data...")
    df = load_multiple_days(batch_dates)
    print(f"✓ Loaded {len(df):,} GPS records")
    
    # Preprocess
    print(f"\n[2/3] Preprocessing...")
    training_data = preprocess_data(df, output_path=output_file)
    
    # Free memory
    del df
    gc.collect()
    
    print(f"\n[3/3] Batch complete!")
    print(f"✓ Generated {len(training_data):,} training examples")
    print(f"✓ Saved to: {output_file}")
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    
    return len(training_data)


def combine_batches(batch_files, output_file):
    """
    Combine multiple preprocessed CSV files
    
    Args:
        batch_files: List of CSV filenames
        output_file: Combined output filename
    
    Returns:
        Combined DataFrame
    """
    print("\n" + "="*80)
    print("COMBINING BATCHES")
    print("="*80)
    
    dfs = []
    for i, file in enumerate(batch_files, 1):
        print(f"[{i}/{len(batch_files)}] Loading {file}...")
        df = pd.read_csv(file)
        print(f"  ✓ {len(df):,} examples")
        dfs.append(df)
    
    print(f"\nCombining...")
    combined = pd.concat(dfs, ignore_index=True)
    
    print(f"✓ Total: {len(combined):,} training examples")
    print(f"✓ Saving to: {output_file}")
    combined.to_csv(output_file, index=False)
    
    return combined


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def train_on_full_dataset():
    """
    Complete pipeline: batch process all data, combine, and train
    """
    print("\n" + "="*80)
    print("🚀 FULL DATASET TRAINING PIPELINE (18 DAYS)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Process batches
    print("\n" + "="*80)
    print("STEP 1: BATCH PROCESSING")
    print("="*80)
    
    batch_files = []
    total_examples = 0
    
    # Batch 1: Nov 3-9
    examples_1 = process_batch(BATCH_1, "BATCH 1 (Nov 3-9)", "training_batch1.csv")
    batch_files.append("training_batch1.csv")
    total_examples += examples_1
    
    # Batch 2: Nov 10-15
    examples_2 = process_batch(BATCH_2, "BATCH 2 (Nov 10-15)", "training_batch2.csv")
    batch_files.append("training_batch2.csv")
    total_examples += examples_2
    
    # Batch 3: Nov 16-19
    examples_3 = process_batch(BATCH_3, "BATCH 3 (Nov 16-19)", "training_batch3.csv")
    batch_files.append("training_batch3.csv")
    total_examples += examples_3
    
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Batch 1: {examples_1:,} examples")
    print(f"Batch 2: {examples_2:,} examples")
    print(f"Batch 3: {examples_3:,} examples")
    print(f"Total: {total_examples:,} examples")
    
    # Step 2: Combine batches
    print("\n" + "="*80)
    print("STEP 2: COMBINING BATCHES")
    print("="*80)
    
    combined_data = combine_batches(batch_files, "training_full_18days.csv")
    
    # Step 3: Train model
    print("\n" + "="*80)
    print("STEP 3: TRAINING FINAL MODEL")
    print("="*80)
    
    model, metrics, encoders = train_and_evaluate_xgboost(
        combined_data,
        save_model_path='eta_model_18days.pkl',
        save_plot_path='model_evaluation_18days.png'
    )
    
    # Final summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 Final Model Performance:")
    print(f"  Training examples: {len(combined_data):,}")
    print(f"  Test MAE: {metrics['test_mae_min']:.2f} minutes")
    print(f"  Test RMSE: {metrics['test_rmse_min']:.2f} minutes")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Accuracy ±5 min: {metrics['accuracy_5min']:.1f}%")
    print(f"\n📁 Outputs:")
    print(f"  - Training data: training_full_18days.csv")
    print(f"  - Model: eta_model_18days.pkl")
    print(f"  - Plots: model_evaluation_18days.png")
    
    return model, metrics, encoders


# ============================================================================
# ALTERNATIVE: INCREMENTAL TRAINING (if combined data too large)
# ============================================================================

def train_incrementally():
    """
    Alternative approach: Train on batches incrementally
    
    XGBoost supports incremental training with xgb_model parameter
    """
    print("\n" + "="*80)
    print("🚀 INCREMENTAL TRAINING PIPELINE")
    print("="*80)
    
    model = None
    
    for i, (batch_dates, batch_name) in enumerate([
        (BATCH_1, "Batch 1 (Nov 3-9)"),
        (BATCH_2, "Batch 2 (Nov 10-15)"),
        (BATCH_3, "Batch 3 (Nov 16-19)")
    ], 1):
        print(f"\n[{i}/3] Training on {batch_name}...")
        
        # Load preprocessed batch
        batch_file = f"training_batch{i}.csv"
        df = pd.read_csv(batch_file)
        
        # Train (or continue training)
        if model is None:
            # First batch: train from scratch
            model, metrics, encoders = train_and_evaluate_xgboost(df)
        else:
            # Subsequent batches: continue training
            # Note: This requires modifying train_and_evaluate_xgboost
            # to support xgb_model parameter
            print("  ⚠️  Incremental training not yet implemented")
            print("  ⚠️  Use combine_batches approach instead")
            break
        
        del df
        gc.collect()
    
    return model, metrics, encoders


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Run full pipeline
    model, metrics, encoders = train_on_full_dataset()
    
    print("\n✓ Training complete! Model ready for deployment.")
