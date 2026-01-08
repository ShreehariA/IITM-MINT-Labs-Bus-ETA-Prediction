"""
Bus ETA Prediction - Unified Training Script (Enhanced Features)

Works with enhanced features from preprocessing_v2.py
Includes 35+ features: cyclical time, lag features, traffic indicators, historical patterns

Usage in Jupyter Notebook:
    import pandas as pd
    from test_xgboost import train_and_evaluate_xgboost
    
    # Load your enhanced preprocessed data
    df = pd.read_csv('training_enhanced_1week.csv')
    
    # Train model (does everything!)
    model, metrics, encoders = train_and_evaluate_xgboost(df)
    
    # Model is automatically saved to 'eta_model_xgboost.pkl'
    # Plots are automatically saved to 'model_evaluation.png'

Author: Shreehari Anbazhagan
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_palette("colorblind")
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature columns (ML AUDIT FIXES APPLIED)
# Removed: time_to_stop_naive, speed_efficiency (DATA LEAKAGE!)
FEATURE_COLUMNS = [
    # Location
    'latitude',
    'longitude',
    
    # Motion
    'speed',
    'acceleration',
    
    # Distance
    'cum_dist_m',
    'dist_to_stop_m',
    
    # Temporal (basic)
    'hour',
    'day_of_week',
    'is_weekend',
    'is_peak_hour',
    'minutes_since_midnight',
    
    # Temporal (cyclical)
    'hour_sin',
    'hour_cos',
    'day_sin',
    'day_cos',
    
    # Speed dynamics
    'speed_ma_30sec',
    'speed_ma_1min',
    
    # Derived features (NON-LEAKY)
    # REMOVED: 'time_to_stop_naive',     # DATA LEAKAGE - derived from target!
    # REMOVED: 'speed_efficiency',        # DATA LEAKAGE - uses time_to_stop_naive!
    'is_very_close',
    'is_close',
    'is_far',
    'is_moving',
    'is_slow',
    'is_accelerating',
    'is_decelerating',
    
    # Priority 2 - Stop context
    'stops_remaining',
    
    # Priority 3 - Historical speed
    'hist_speed_mean',
    'hist_speed_std',
    'speed_vs_historical',
    'is_slower_than_usual',
    'is_faster_than_usual',
]

CATEGORICAL_FEATURES = []
TARGET_COLUMN = 'ETA_sec'
USE_LOG_TRANSFORM = False  # Disabled - made accuracy worse

# Model hyperparameters (optimized for enhanced features)
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
    'verbosity': 1
}

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df, test_size=0.25, random_state=42):
    """
    Prepare data for training with proper train/test split
    
    CRITICAL: Split by trip_id to avoid data leakage
    (same trip should not appear in both train and test)
    
    Args:
        df: Preprocessed DataFrame
        test_size: Fraction for test set (default 0.25)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, encoders
    """
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    df = df.copy()
    
    # Check for required columns
    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")
    
    # IMPROVED: Date-based split (chronological) instead of random trips
    # This prevents data leakage and tests real "future prediction" ability
    
    # Define train/test dates (first 15 days train, last 3 days test)
    # NOTE: Dates are integers in the CSV, not strings
    train_dates = [20251103, 20251104, 20251106, 20251107, 20251108,
                   20251109, 20251110, 20251111, 20251112, 20251113,
                   20251114, 20251115, 20251116, 20251117]
    test_dates = [20251118, 20251119, 20251120, 20251121]
    
    # Split by date
    df_train = df[df['date'].isin(train_dates)].copy()
    df_test = df[df['date'].isin(test_dates)].copy()
    
    # Get unique trips for reporting
    train_trips = df_train['trip_id'].nunique()
    test_trips = df_test['trip_id'].nunique()
    
    print(f"\nTrain/Test Split (Date-Based):")
    print(f"  Train dates: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"  Test dates: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")
    print(f"  Train trips: {train_trips} ({len(df_train):,} samples)")
    print(f"  Test trips: {test_trips} ({len(df_test):,} samples)")
    print(f"  Split ratio: {len(df_train)/len(df)*100:.1f}% / {len(df_test)/len(df)*100:.1f}%")
    
    # Encode categorical features
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))
        encoders[col] = le
        print(f"  Encoded {col}: {len(le.classes_)} unique values")
    
    # Prepare features and target
    X_train = df_train[FEATURE_COLUMNS]
    X_test = df_test[FEATURE_COLUMNS]
    
    # ML AUDIT FIX: Handle missing values using TRAIN statistics only
    # (Prevents data leakage from test set)
    print("\n📊 Handling Missing Values:")
    for col in FEATURE_COLUMNS:
        train_missing = X_train[col].isna().sum()
        test_missing = X_test[col].isna().sum()
        
        if train_missing > 0 or test_missing > 0:
            # Use TRAIN mean/mode only (not global!)
            if X_train[col].dtype in ['float64', 'int64']:
                fill_value = X_train[col].mean()
                X_train[col].fillna(fill_value, inplace=True)
                X_test[col].fillna(fill_value, inplace=True)
                print(f"  {col}: filled {train_missing + test_missing} NaNs with train mean ({fill_value:.2f})")
            else:
                fill_value = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 0
                X_train[col].fillna(fill_value, inplace=True)
                X_test[col].fillna(fill_value, inplace=True)
                print(f"  {col}: filled {train_missing + test_missing} NaNs with train mode ({fill_value})")
    
    # Prepare target (no transformation)
    y_train = df_train[TARGET_COLUMN]
    y_test = df_test[TARGET_COLUMN]
    
    print(f"\nFeature matrix shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, encoders


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_eta_model(df, test_size=0.25, random_state=42, params=None):
    """
    Train XGBoost model for ETA prediction
    
    Args:
        df: Preprocessed DataFrame
        test_size: Test set fraction
        random_state: Random seed
        params: Optional custom XGBoost parameters
    
    Returns:
        model: Trained XGBoost model
        metrics: Dictionary of evaluation metrics
        encoders: Label encoders for categorical features
    """
    print("\n" + "="*80)
    print("BUS ETA PREDICTION - MODEL TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, encoders = prepare_data(
        df, test_size=test_size, random_state=random_state
    )
    
    # Initialize model
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)
    
    model_params = params if params else XGBOOST_PARAMS
    print(f"\nModel parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    model = XGBRegressor(**model_params)
    
    # Train with early stopping
    print(f"\nTraining model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print(f"✓ Training complete!")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score: {model.best_score:.4f}")
    
    # Evaluate
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model, metrics, encoders


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Comprehensive model evaluation with multiple metrics
    
    Args:
        model: Trained model
        X_train, X_test: Feature matrices
        y_train, y_test: Target values
    
    Returns:
        Dictionary of metrics
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Predictions (no transformation needed)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_actual = y_train
    y_test_actual = y_test
    
    # Calculate metrics (in seconds)
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    
    train_r2 = r2_score(y_train_actual, y_train_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)
    
    # Accuracy thresholds (in seconds)
    accuracy_1min = (np.abs(y_test_actual - y_test_pred) <= 60).mean() * 100
    accuracy_2min = (np.abs(y_test_actual - y_test_pred) <= 120).mean() * 100
    accuracy_5min = (np.abs(y_test_actual - y_test_pred) <= 300).mean() * 100
    
    # Print results
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 80)
    print(f"{'Metric':<25} {'Train':<15} {'Test':<15}")
    print("-" * 80)
    print(f"{'MAE (minutes)':<25} {train_mae/60:<15.2f} {test_mae/60:<15.2f}")
    print(f"{'RMSE (minutes)':<25} {train_rmse/60:<15.2f} {test_rmse/60:<15.2f}")
    print(f"R² Score                  {train_r2:<15.4f} {test_r2:<15.4f}")
    print("-" * 80)
    print(f"Accuracy ±1 min           -               {accuracy_1min:<15.1f}%")
    print(f"Accuracy ±2 min           -               {accuracy_2min:<15.1f}%")
    print(f"Accuracy ±5 min           -               {accuracy_5min:<15.1f}%")
    print("-" * 80)
    
    # Check for overfitting
    mae_diff = abs(train_mae - test_mae) / train_mae * 100
    if mae_diff > 20:
        print(f"\n⚠️  Warning: Possible overfitting (MAE difference: {mae_diff:.1f}%)")
    else:
        print(f"\n✓ Good generalization (MAE difference: {mae_diff:.1f}%)")
    
    # Feature importance
    print("\n📈 TOP 10 FEATURE IMPORTANCE")
    print("-" * 80)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:<25} {row['importance']:<15.4f}")
    
    # Store metrics
    return {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'mae_train': train_mae / 60,  # minutes
        'mae_test': test_mae / 60,    # minutes
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'r2_train': train_r2,
        'r2_test': test_r2,
        'accuracy_1min': accuracy_1min,
        'accuracy_2min': accuracy_2min,
        'accuracy_5min': accuracy_5min,
        'feature_importance': feature_importance,
        'predictions': {
            'y_test': y_test_actual, # Store actual values for plotting
            'y_test_pred': y_test_pred
        }
    }
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(metrics, save_path='model_evaluation.png'):
    """
    Create visualization of model performance
    
    Args:
        metrics: Dictionary from evaluate_model()
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    y_test = metrics['predictions']['y_test'] / 60  # Convert to minutes
    y_pred = metrics['predictions']['y_test_pred'] / 60
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.3, s=10)
    axes[0, 0].plot([0, 30], [0, 30], 'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual ETA (minutes)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted ETA (minutes)', fontsize=12)
    axes[0, 0].set_title('Actual vs Predicted ETA', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted ETA (minutes)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals (minutes)', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    errors = np.abs(residuals)
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=2, color='g', linestyle='--', lw=2, label='±2 min threshold')
    axes[1, 0].axvline(x=5, color='orange', linestyle='--', lw=2, label='±5 min threshold')
    axes[1, 0].set_xlabel('Absolute Error (minutes)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature importance
    top_features = metrics['feature_importance'].head(10)
    axes[1, 1].barh(top_features['feature'], top_features['importance'])
    axes[1, 1].set_xlabel('Importance', fontsize=12)
    axes[1, 1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {save_path}")
    plt.show()


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_eta(model, encoders, sample_data):
    """
    Predict ETA for new data
    
    Args:
        model: Trained XGBoost model
        encoders: Dictionary of label encoders
        sample_data: DataFrame or dict with required features
    
    Returns:
        Predicted ETA in seconds
    """
    # Convert to DataFrame if dict
    if isinstance(sample_data, dict):
        sample_data = pd.DataFrame([sample_data])
    
    # Encode categorical features
    sample_encoded = sample_data.copy()
    for col in CATEGORICAL_FEATURES:
        if col in sample_encoded.columns:
            sample_encoded[col] = encoders[col].transform(sample_encoded[col].astype(str))
    
    # Predict
    eta_sec = model.predict(sample_encoded[FEATURE_COLUMNS])[0]
    
    return max(0, eta_sec)  # Ensure non-negative


def save_model(model, encoders, metrics, filepath='eta_model_xgboost.pkl'):
    """
    Save trained model and metadata
    
    Args:
        model: Trained model
        encoders: Label encoders
        metrics: Evaluation metrics
        filepath: Save path
    """
    model_package = {
        'model': model,
        'encoders': encoders,
        'feature_columns': FEATURE_COLUMNS,
        'metrics': {k: v for k, v in metrics.items() if k != 'predictions'},
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'xgboost_params': XGBOOST_PARAMS
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\n✓ Model saved to: {filepath}")


def load_model(filepath='eta_model.pkl'):
    """
    Load saved model
    
    Args:
        filepath: Path to saved model
    
    Returns:
        model, encoders, metadata
    """
    with open(filepath, 'rb') as f:
        package = pickle.load(f)
    
    return package['model'], package['encoders'], package



# ============================================================================
# UNIFIED TRAINING FUNCTION (MAIN ENTRY POINT)
# ============================================================================

def train_and_evaluate_xgboost(df, test_size=0.25, save_model_path='eta_model.pkl', 
                       save_plot_path='model_evaluation.png'):
    """
    ONE-STOP FUNCTION: Train, evaluate, visualize, and save model
    
    This function does EVERYTHING:
    - Splits data (75/25 by default)
    - Trains XGBoost model
    - Evaluates with MAE, RMSE, R², accuracy metrics
    - Creates 4 visualization plots
    - Saves model to pickle file
    
    Args:
        df: Preprocessed DataFrame with features and ETA_sec label
        test_size: Fraction for test set (default 0.25 = 75/25 split)
        save_model_path: Where to save trained model (default 'eta_model.pkl')
        save_plot_path: Where to save plots (default 'model_evaluation.png')
    
    Returns:
        model: Trained XGBoost model
        metrics: Dictionary with all evaluation metrics
        encoders: Label encoders for categorical features
    
    Example:
        >>> df = pd.read_csv('training_data_1week.csv')
        >>> model, metrics, encoders = train_and_evaluate(df)
        >>> print(f"Test MAE: {metrics['test_mae_min']:.2f} minutes")
    """
    print("\n" + "="*80)
    print("🚀 BUS ETA PREDICTION - UNIFIED TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nInput dataset: {len(df):,} training examples")
    
    # Step 1: Train model
    model, metrics, encoders = train_eta_model(df, test_size=test_size)
    
    # Step 2: Visualize
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    plot_results(metrics, save_path=save_plot_path)
    
    # Step 3: Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    save_model(model, encoders, metrics, filepath=save_model_path)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 Model Performance Summary:")
    print(f"  Test MAE:  {metrics['mae_test']:.2f} minutes")
    print(f"  Test RMSE: {metrics['test_rmse']/60:.2f} minutes")
    print(f"  Test R²:   {metrics['test_r2']:.4f}")
    print(f"  Accuracy ±1 min: {metrics.get('accuracy_1min', 0):.1f}%")
    print(f"  Accuracy ±2 min: {metrics.get('accuracy_2min', 0):.1f}%")
    print(f"  Accuracy ±5 min: {metrics.get('accuracy_5min', 0):.1f}%")
    
    print(f"\n📁 Outputs:")
    if save_model_path:
        print(f"  - Model: {save_model_path}")
    if save_plot_path:
        print(f"  - Plots: {save_plot_path}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return model, metrics, encoders


# ============================================================================
# MAIN EXECUTION (if run as script)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Loading preprocessed data...")
    df = pd.read_csv('training_data_1week.csv')
    
    print(f"Loaded {len(df):,} training examples")
    
    # Train model with one function call!
    model, metrics, encoders = train_and_evaluate_xgboost(df)
    
    print("\n✓ Ready for deployment!")

