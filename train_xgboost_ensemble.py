"""
XGBoost Model - First ensemble component
Optimized for precise ±1 min predictions
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import torch

print("="*80)
print("TRAINING XGBOOST FOR ENSEMBLE")
print("="*80)

# 1. Load graphs and convert to tabular format
print("\n📂 Loading graph dataset...")
graphs = torch.load('bus_graphs_full.pt', weights_only=False)
print(f"  Total graphs: {len(graphs)}")

# 2. Extract features from graphs
print("\n🔄 Converting graphs to tabular format...")

def graph_to_features(graph):
    """Extract comprehensive features from graph"""
    x = graph.x.numpy()
    
    # Node-level statistics (aggregated)
    node_mean = x.mean(axis=0)
    node_std = x.std(axis=0)
    node_max = x.max(axis=0)
    node_min = x.min(axis=0)
    node_median = np.median(x, axis=0)
    
    # Graph-level features
    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.shape[1]
    
    # Target node features (where we're predicting)
    if hasattr(graph, 'target_node'):
        target_idx = graph.target_node
        if target_idx < len(x):
            target_features = x[target_idx]
        else:
            target_features = node_mean
    else:
        target_features = node_mean
    
    # Advanced features
    # Speed variance (traffic variability)
    speed_variance = x[:, 0].var() if len(x) > 0 else 0
    
    # Distance progression (how far through route)
    if len(x) > 1:
        dist_progression = x[:, 1].mean() / (x[:, 1].max() + 1e-6)
    else:
        dist_progression = 0
    
    # Time features
    hour_features = x[:, 2] if len(x) > 0 else np.array([0])
    is_rush_hour = ((hour_features >= 7) & (hour_features <= 9) | 
                    (hour_features >= 17) & (hour_features <= 19)).mean()
    
    # Combine all features
    features = np.concatenate([
        node_mean,      # 8 features
        node_std,       # 8 features
        node_max,       # 8 features
        node_min,       # 8 features
        node_median,    # 8 features
        target_features,# 8 features
        [num_nodes, num_edges, speed_variance, dist_progression, is_rush_hour]  # 5 features
    ])
    
    return features

# Convert all graphs
print("  Converting graphs...")
X_all = []
y_all = []

for g in graphs:
    features = graph_to_features(g)
    X_all.append(features)
    y_all.append(g.y.item())

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"  Feature matrix: {X_all.shape}")
print(f"  Total features: {X_all.shape[1]}")

# 3. Random split (same as GNN)
np.random.seed(42)
indices = np.random.permutation(len(X_all))
X_all = X_all[indices]
y_all = y_all[indices]

train_size = int(0.7 * len(X_all))
val_size = int(0.15 * len(X_all))

X_train = X_all[:train_size]
y_train = y_all[:train_size]

X_val = X_all[train_size:train_size+val_size]
y_val = y_all[train_size:train_size+val_size]

X_test = X_all[train_size+val_size:]
y_test = y_all[train_size+val_size:]

print(f"\n📊 Dataset split:")
print(f"  Train: {len(X_train)}")
print(f"  Val:   {len(X_val)}")
print(f"  Test:  {len(X_test)}")

# 4. Normalize features
print("\n📊 Normalizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save scaler
with open('xgb_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 5. Train XGBoost with aggressive parameters
print("\n🚀 Training XGBoost...")

xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# 6. Evaluate
print("\n📊 Evaluating XGBoost...")

train_pred = xgb_model.predict(X_train)
val_pred = xgb_model.predict(X_val)
test_pred = xgb_model.predict(X_test)

# Calculate metrics
def calculate_metrics(y_true, y_pred, name):
    mae = np.mean(np.abs(y_true - y_pred)) / 60
    rmse = np.sqrt(np.mean((y_true - y_pred)**2)) / 60
    acc_1min = (np.abs(y_true - y_pred) <= 60).mean() * 100
    acc_2min = (np.abs(y_true - y_pred) <= 120).mean() * 100
    acc_5min = (np.abs(y_true - y_pred) <= 300).mean() * 100
    
    print(f"\n{name} Results:")
    print(f"  MAE:  {mae:.2f} minutes")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  ±1 min: {acc_1min:.1f}%")
    print(f"  ±2 min: {acc_2min:.1f}%")
    print(f"  ±5 min: {acc_5min:.1f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'acc_1min': acc_1min,
        'acc_2min': acc_2min,
        'acc_5min': acc_5min
    }

train_metrics = calculate_metrics(y_train, train_pred, "Train")
val_metrics = calculate_metrics(y_val, val_pred, "Validation")
test_metrics = calculate_metrics(y_test, test_pred, "Test")

# 7. Save model
print("\n💾 Saving XGBoost model...")
xgb_model.save_model('xgboost_model.json')

print("\n" + "="*80)
print("✅ XGBOOST TRAINING COMPLETE!")
print("="*80)
print(f"\nTest Performance:")
print(f"  ±1 min: {test_metrics['acc_1min']:.1f}%")
print(f"  ±2 min: {test_metrics['acc_2min']:.1f}%")
print(f"  ±5 min: {test_metrics['acc_5min']:.1f}%")
