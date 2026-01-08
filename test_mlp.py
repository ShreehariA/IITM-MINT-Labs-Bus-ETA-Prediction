"""
Bus ETA Prediction - MLP Training Script

Works with enhanced features from preprocessing_v2.py
Uses sklearn's MLPRegressor (Multi-Layer Perceptron) for comparison with XGBoost

Usage in Jupyter Notebook:
    import pandas as pd
    from test_mlp import train_and_evaluate_mlp
    
    # Load your enhanced preprocessed data
    df = pd.read_csv('training_enhanced_1week.csv')
    
    # Train model (does everything!)
    model, metrics, scalers = train_and_evaluate_mlp(df)
    
    # Model is automatically saved to 'eta_model_mlp.pkl'
    # Plots are automatically saved to 'model_evaluation_mlp.png'

Author: Shreehari Anbazhagan
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
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

# Feature columns (same as test_xgboost.py)
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
    
    # Original temporal features
    'hour',
    'day_of_week',
    'is_weekend',
    'is_peak_hour',
    'minutes_since_midnight',
    
    # Cyclical time features
    'hour_sin',
    'hour_cos',
    'day_sin',
    'day_cos',
    
    # Speed dynamics
    'speed_ma_1min',
    'speed_ma_2min',
    
    # Lag features
    'speed_lag_1',
    'speed_lag_5',
    'speed_lag_10',
    'dist_lag_1',
    
    # Traffic indicators
    'speed_variance_5min',
    'time_below_5kmh_pct',
    'num_speed_drops',
    'congestion_score',
    
    # Improved stop features
    'time_since_last_movement',
    'num_stops_last_5min',
    
    # Historical features
    'route_hour_avg_speed',
    'route_weekend_avg_speed',
    'route_avg_cum_dist',
    
    # Route context
    'route_id',  # Categorical
    'stop_id',   # Categorical
    'stop_sequence',
]

CATEGORICAL_FEATURES = ['route_id', 'stop_id']
TARGET_COLUMN = 'ETA_sec'

# MLP hyperparameters
MLP_PARAMS = {
    'hidden_layer_sizes': (128, 64, 32),  # 3 hidden layers
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,  # L2 regularization
    'batch_size': 256,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 200,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 20,
    'random_state': 42,
    'verbose': True
}

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df, test_size=0.25, random_state=42):
    """
    Prepare data for MLP training with proper scaling
    
    CRITICAL: 
    - Split by trip_id to avoid data leakage
    - Scale features (MLPs need normalized inputs!)
    
    Args:
        df: Preprocessed DataFrame
        test_size: Fraction for test set (default 0.25)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, encoders, scaler
    """
    print("\n" + "="*80)
    print("DATA PREPARATION FOR MLP")
    print("="*80)
    
    df = df.copy()
    
    # Check for required columns
    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_COLUMN}")
    
    # Split by trip_id to avoid leakage
    unique_trips = df['trip_id'].unique()
    train_trips, test_trips = train_test_split(
        unique_trips, 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_mask = df['trip_id'].isin(train_trips)
    test_mask = df['trip_id'].isin(test_trips)
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"\nTrain/Test Split:")
    print(f"  Train trips: {len(train_trips)} ({len(df_train):,} samples)")
    print(f"  Test trips: {len(test_trips)} ({len(df_test):,} samples)")
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
    y_train = df_train[TARGET_COLUMN]
    y_test = df_test[TARGET_COLUMN]
    
    # CRITICAL: Scale features for MLP
    print(f"\n⚠️  Scaling features (required for MLP)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for consistency
    X_train = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS, index=X_test.index)
    
    print(f"  ✓ Features scaled to mean=0, std=1")
    
    print(f"\nFeature matrix shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, encoders, scaler


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_eta_model(df, test_size=0.25, random_state=42, params=None):
    """
    Train MLP model for ETA prediction
    
    Args:
        df: Preprocessed DataFrame
        test_size: Test set fraction
        random_state: Random seed
        params: Optional custom MLP parameters
    
    Returns:
        model: Trained MLP model
        metrics: Dictionary of evaluation metrics
        encoders: Label encoders for categorical features
        scaler: StandardScaler for features
    """
    print("\n" + "="*80)
    print("BUS ETA PREDICTION - MLP TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, encoders, scaler = prepare_data(
        df, test_size=test_size, random_state=random_state
    )
    
    # Initialize model
    print("\n" + "="*80)
    print("TRAINING MLP MODEL")
    print("="*80)
    
    model_params = params if params else MLP_PARAMS
    print(f"\nModel architecture:")
    print(f"  Hidden layers: {model_params['hidden_layer_sizes']}")
    print(f"  Activation: {model_params['activation']}")
    print(f"  Solver: {model_params['solver']}")
    print(f"  Learning rate: {model_params['learning_rate_init']}")
    print(f"  Max iterations: {model_params['max_iter']}")
    
    model = MLPRegressor(**model_params)
    
    # Train
    print(f"\nTraining model...")
    print(f"(This may take several minutes...)")
    model.fit(X_train, y_train)
    
    print(f"\n✓ Training complete!")
    print(f"  Iterations: {model.n_iter_}")
    print(f"  Final loss: {model.loss_:.4f}")
    
    # Evaluate
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model, metrics, encoders, scaler


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
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics (in seconds)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Calculate accuracy within thresholds
    test_errors = np.abs(y_test - y_test_pred)
    acc_2min = (test_errors <= 120).mean() * 100
    acc_5min = (test_errors <= 300).mean() * 100
    
    # Print results
    print("\n📊 PERFORMANCE METRICS")
    print("-" * 80)
    print(f"{'Metric':<25} {'Train':<15} {'Test':<15}")
    print("-" * 80)
    print(f"{'MAE (minutes)':<25} {train_mae/60:<15.2f} {test_mae/60:<15.2f}")
    print(f"{'RMSE (minutes)':<25} {train_rmse/60:<15.2f} {test_rmse/60:<15.2f}")
    print(f"{'R² Score':<25} {train_r2:<15.4f} {test_r2:<15.4f}")
    print("-" * 80)
    print(f"{'Accuracy ±2 min':<25} {'-':<15} {acc_2min:<15.1f}%")
    print(f"{'Accuracy ±5 min':<25} {'-':<15} {acc_5min:<15.1f}%")
    print("-" * 80)
    
    # Check for overfitting
    mae_diff = abs(train_mae - test_mae) / train_mae * 100
    if mae_diff > 20:
        print(f"\n⚠️  Warning: Possible overfitting (MAE difference: {mae_diff:.1f}%)")
    else:
        print(f"\n✓ Good generalization (MAE difference: {mae_diff:.1f}%)")
    
    # Store metrics
    metrics = {
        'train_mae_sec': train_mae,
        'test_mae_sec': test_mae,
        'train_mae_min': train_mae / 60,
        'test_mae_min': test_mae / 60,
        'train_rmse_sec': train_rmse,
        'test_rmse_sec': test_rmse,
        'train_rmse_min': train_rmse / 60,
        'test_rmse_min': test_rmse / 60,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'accuracy_2min': acc_2min,
        'accuracy_5min': acc_5min,
        'predictions': {
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
    }
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(metrics, save_path='model_evaluation_mlp.png'):
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
    axes[0, 0].set_title('MLP: Actual vs Predicted ETA', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted ETA (minutes)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals (minutes)', fontsize=12)
    axes[0, 1].set_title('MLP: Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    errors = np.abs(residuals)
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=2, color='g', linestyle='--', lw=2, label='±2 min threshold')
    axes[1, 0].axvline(x=5, color='orange', linestyle='--', lw=2, label='±5 min threshold')
    axes[1, 0].set_xlabel('Absolute Error (minutes)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('MLP: Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Learning curve (loss over iterations)
    axes[1, 1].text(0.5, 0.5, 'MLP Neural Network\n\nNo feature importance\navailable for MLPs\n\n(Use permutation importance\nif needed)', 
                    ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Note: Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {save_path}")
    plt.show()


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_eta(model, encoders, scaler, sample_data):
    """
    Predict ETA for new data
    
    Args:
        model: Trained MLP model
        encoders: Dictionary of label encoders
        scaler: StandardScaler for features
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
    
    # Scale features
    sample_scaled = scaler.transform(sample_encoded[FEATURE_COLUMNS])
    
    # Predict
    eta_sec = model.predict(sample_scaled)[0]
    
    return max(0, eta_sec)  # Ensure non-negative


def save_model(model, encoders, scaler, metrics, filepath='eta_model_mlp.pkl'):
    """
    Save trained model and metadata
    
    Args:
        model: Trained model
        encoders: Label encoders
        scaler: StandardScaler
        metrics: Evaluation metrics
        filepath: Save path
    """
    model_package = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'feature_columns': FEATURE_COLUMNS,
        'metrics': {k: v for k, v in metrics.items() if k != 'predictions'},
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mlp_params': MLP_PARAMS
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\n✓ Model saved to: {filepath}")


def load_model(filepath='eta_model_mlp.pkl'):
    """
    Load saved model
    
    Args:
        filepath: Path to saved model
    
    Returns:
        model, encoders, scaler, metadata
    """
    with open(filepath, 'rb') as f:
        package = pickle.load(f)
    
    return package['model'], package['encoders'], package['scaler'], package


# ============================================================================
# UNIFIED TRAINING FUNCTION (MAIN ENTRY POINT)
# ============================================================================

def train_and_evaluate_mlp(df, test_size=0.25, save_model_path='eta_model_mlp.pkl', 
                           save_plot_path='model_evaluation_mlp.png'):
    """
    ONE-STOP FUNCTION: Train, evaluate, visualize, and save MLP model
    
    This function does EVERYTHING:
    - Splits data (75/25 by default)
    - Scales features (required for MLP!)
    - Trains MLP model
    - Evaluates with MAE, RMSE, R², accuracy metrics
    - Creates 4 visualization plots
    - Saves model to pickle file
    
    Args:
        df: Preprocessed DataFrame with features and ETA_sec label
        test_size: Fraction for test set (default 0.25 = 75/25 split)
        save_model_path: Where to save trained model (default 'eta_model_mlp.pkl')
        save_plot_path: Where to save plots (default 'model_evaluation_mlp.png')
    
    Returns:
        model: Trained MLP model
        metrics: Dictionary with all evaluation metrics
        encoders: Label encoders for categorical features
        scaler: StandardScaler for features
    
    Example:
        >>> df = pd.read_csv('training_enhanced_1week.csv')
        >>> model, metrics, encoders, scaler = train_and_evaluate_mlp(df)
        >>> print(f"Test MAE: {metrics['test_mae_min']:.2f} minutes")
    """
    print("\n" + "="*80)
    print("🧠 BUS ETA PREDICTION - MLP TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nInput dataset: {len(df):,} training examples")
    
    # Step 1: Train model
    model, metrics, encoders, scaler = train_eta_model(df, test_size=test_size)
    
    # Step 2: Visualize
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    plot_results(metrics, save_path=save_plot_path)
    
    # Step 3: Save model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    save_model(model, encoders, scaler, metrics, filepath=save_model_path)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 Model Performance Summary:")
    print(f"  Test MAE:  {metrics['test_mae_min']:.2f} minutes")
    print(f"  Test RMSE: {metrics['test_rmse_min']:.2f} minutes")
    print(f"  Test R²:   {metrics['test_r2']:.4f}")
    print(f"  Accuracy ±2 min: {metrics['accuracy_2min']:.1f}%")
    print(f"  Accuracy ±5 min: {metrics['accuracy_5min']:.1f}%")
    print(f"\n📁 Outputs:")
    print(f"  - Model: {save_model_path}")
    print(f"  - Plots: {save_plot_path}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model, metrics, encoders, scaler


# ============================================================================
# MAIN EXECUTION (if run as script)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Loading preprocessed data...")
    df = pd.read_csv('training_enhanced_1week.csv')
    
    print(f"Loaded {len(df):,} training examples")
    
    # Train model with one function call!
    model, metrics, encoders, scaler = train_and_evaluate_mlp(df)
    
    print("\n✓ Ready for deployment!")
