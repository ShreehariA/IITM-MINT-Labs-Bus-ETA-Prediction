"""
LightGBM Model - Second ensemble component
Faster training, different tree structure than XGBoost
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pickle
import torch

print("="*80)
print("TRAINING LIGHTGBM FOR ENSEMBLE")
print("="*80)

# 1. Load graphs and convert to tabular format
print("\n📂 Loading graph dataset...")
graphs = torch.load('bus_graphs_full.pt', weights_only=False)

def graph_to_features(graph):
    """Extract features from graph"""
    x = graph.x.numpy()
    
    node_mean = x.mean(axis=0)
    node_std = x.std(axis=0)
    node_max = x.max(axis=0)
    node_min = x.min(axis=0)
    node_median = np.median(x, axis=0)
    
    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.shape[1]
    
    if hasattr(graph, 'target_node') and graph.target_node < len(x):
        target_features = x[graph.target_node]
    else:
        target_features = node_mean
    
    speed_variance = x[:, 0].var() if len(x) > 0 else 0
    dist_progression = x[:, 1].mean() / (x[:, 1].max() + 1e-6) if len(x) > 1 else 0
    
    hour_features = x[:, 2] if len(x) > 0 else np.array([0])
    is_rush_hour = ((hour_features >= 7) & (hour_features <= 9) | 
                    (hour_features >= 17) & (hour_features <= 19)).mean()
    
    features = np.concatenate([
        node_mean, node_std, node_max, node_min, node_median,
        target_features,
        [num_nodes, num_edges, speed_variance, dist_progression, is_rush_hour]
    ])
    
    return features

# Convert graphs
print("\n🔄 Converting graphs...")
X_all = np.array([graph_to_features(g) for g in graphs])
y_all = np.array([g.y.item() for g in graphs])

# Random split
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

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

with open('lgb_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train LightGBM
print("\n🚀 Training LightGBM...")

lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    num_leaves=127,
    learning_rate=0.01,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    max_depth=10,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=100
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Evaluate
print("\n📊 Evaluating LightGBM...")

test_pred = lgb_model.predict(X_test)

mae = np.mean(np.abs(y_test - test_pred)) / 60
rmse = np.sqrt(np.mean((y_test - test_pred)**2)) / 60
acc_1min = (np.abs(y_test - test_pred) <= 60).mean() * 100
acc_2min = (np.abs(y_test - test_pred) <= 120).mean() * 100
acc_5min = (np.abs(y_test - test_pred) <= 300).mean() * 100

print(f"\n📈 Test Results:")
print(f"  MAE:  {mae:.2f} minutes")
print(f"  RMSE: {rmse:.2f} minutes")
print(f"  ±1 min: {acc_1min:.1f}%")
print(f"  ±2 min: {acc_2min:.1f}%")
print(f"  ±5 min: {acc_5min:.1f}%")

# Save
lgb_model.booster_.save_model('lightgbm_model.txt')

print("\n✅ LIGHTGBM COMPLETE!")
print("="*80)
