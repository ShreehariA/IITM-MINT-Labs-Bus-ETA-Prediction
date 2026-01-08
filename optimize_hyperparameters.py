"""
Hyperparameter optimization using Optuna
Find optimal XGBoost parameters to push ±1 min accuracy higher
"""
import optuna
import xgboost as xgb
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pickle

print("="*80)
print("HYPERPARAMETER OPTIMIZATION FOR 80% TARGET")
print("="*80)

# Load data (reuse feature extraction from original)
graphs = torch.load('bus_graphs_full.pt', weights_only=False)

def graph_to_features(graph):
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

print("\n🔄 Extracting features...")
X_all = np.array([graph_to_features(g) for g in graphs])
y_all = np.array([g.y.item() for g in graphs])

# Split
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

# Optuna objective
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    val_pred = model.predict(X_val)
    acc_1min = (np.abs(y_val - val_pred) <= 60).mean() * 100
    
    return acc_1min

# Optimize
print("\n🔍 Running Optuna optimization (20 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

print(f"\n✅ Best ±1 min accuracy: {study.best_value:.1f}%")
print(f"\n📋 Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best params
print("\n🚀 Training final model with best parameters...")
best_model = xgb.XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Evaluate on test
test_pred = best_model.predict(X_test)

mae = np.mean(np.abs(y_test - test_pred)) / 60
rmse = np.sqrt(np.mean((y_test - test_pred)**2)) / 60
acc_1min = (np.abs(y_test - test_pred) <= 60).mean() * 100
acc_2min = (np.abs(y_test - test_pred) <= 120).mean() * 100
acc_5min = (np.abs(y_test - test_pred) <= 300).mean() * 100

print(f"\n📈 Optimized Test Results:")
print(f"  MAE:  {mae:.2f} minutes")
print(f"  RMSE: {rmse:.2f} minutes")
print(f"  ±1 min: {acc_1min:.1f}%")
print(f"  ±2 min: {acc_2min:.1f}%")
print(f"  ±5 min: {acc_5min:.1f}%")

# Save
best_model.save_model('xgboost_optimized.json')

print("\n✅ OPTIMIZATION COMPLETE!")
print("="*80)
