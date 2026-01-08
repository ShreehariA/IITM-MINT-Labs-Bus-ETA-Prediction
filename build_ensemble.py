"""
Stacking Ensemble - Combines GNN + XGBoost + LightGBM
Uses meta-learner to optimally weight predictions
"""
import torch
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from gnn_temporal_attention import TemporalGraphAttention
from torch_geometric.loader import DataLoader

print("="*80)
print("BUILDING STACKING ENSEMBLE")
print("="*80)

# 1. Load all models
print("\n📂 Loading trained models...")

# GNN
gnn_model = TemporalGraphAttention(num_features=8, hidden=128, heads=8, dropout=0.3)
gnn_model.load_state_dict(torch.load('tgat_best_random.pt', weights_only=False))
gnn_model.eval()

# XGBoost
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('xgboost_model.json')

# LightGBM
lgb_model = lgb.Booster(model_file='lightgbm_model.txt')

# Scalers
with open('scalers_random.pkl', 'rb') as f:
    gnn_scalers = pickle.load(f)

with open('xgb_scaler.pkl', 'rb') as f:
    xgb_scaler = pickle.load(f)

with open('lgb_scaler.pkl', 'rb') as f:
    lgb_scaler = pickle.load(f)

print("  ✓ All models loaded!")

# 2. Load graphs
print("\n📂 Loading graph dataset...")
graphs = torch.load('bus_graphs_full.pt', weights_only=False)

# Random split (same seed)
np.random.seed(42)
import random
random.seed(42)
random.shuffle(graphs)

train_size = int(0.7 * len(graphs))
val_size = int(0.15 * len(graphs))

train_graphs = graphs[:train_size]
val_graphs = graphs[train_size:train_size+val_size]
test_graphs = graphs[train_size+val_size:]

# 3. Generate predictions from all base models
print("\n🔮 Generating base model predictions...")

def graph_to_xgb_features(graph):
    """Convert graph to XGBoost features"""
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

def get_ensemble_predictions(graphs_list, split_name):
    """Get predictions from all 3 models"""
    print(f"  Predicting {split_name}...")
    
    # GNN predictions
    loader = DataLoader(graphs_list, batch_size=32)
    gnn_preds = []
    
    with torch.no_grad():
        for batch in loader:
            pred = gnn_model(batch)
            # Denormalize
            pred = pred.cpu().numpy() * gnn_scalers['target_range'] + gnn_scalers['target_min']
            gnn_preds.extend(pred)
    
    gnn_preds = np.array(gnn_preds)
    
    # XGBoost predictions
    xgb_features = np.array([graph_to_xgb_features(g) for g in graphs_list])
    xgb_features = xgb_scaler.transform(xgb_features)
    xgb_preds = xgb_model.predict(xgb_features)
    
    # LightGBM predictions
    lgb_features = xgb_features  # Same features
    lgb_preds = lgb_model.predict(lgb_features)
    
    # Get actual targets
    actuals = np.array([g.y.item() for g in graphs_list])
    
    return gnn_preds, xgb_preds, lgb_preds, actuals

# Get predictions for all splits
train_gnn, train_xgb, train_lgb, train_y = get_ensemble_predictions(train_graphs, "train")
val_gnn, val_xgb, val_lgb, val_y = get_ensemble_predictions(val_graphs, "val")
test_gnn, test_xgb, test_lgb, test_y = get_ensemble_predictions(test_graphs, "test")

# 4. Train meta-learner (Ridge regression)
print("\n🧠 Training meta-learner...")

# Stack predictions as features
X_train_meta = np.column_stack([train_gnn, train_xgb, train_lgb])
X_val_meta = np.column_stack([val_gnn, val_xgb, val_lgb])
X_test_meta = np.column_stack([test_gnn, test_xgb, test_lgb])

# Train Ridge regression as meta-learner
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_train_meta, train_y)

print(f"  Meta-model weights:")
print(f"    GNN:      {meta_model.coef_[0]:.3f}")
print(f"    XGBoost:  {meta_model.coef_[1]:.3f}")
print(f"    LightGBM: {meta_model.coef_[2]:.3f}")
print(f"    Intercept: {meta_model.intercept_:.1f}")

# 5. Evaluate ensemble
print("\n📊 Evaluating Stacking Ensemble...")

def evaluate_predictions(y_true, y_pred, name):
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
    
    return acc_1min, acc_2min, acc_5min

# Ensemble predictions
test_ensemble = meta_model.predict(X_test_meta)

# Compare individual models vs ensemble
print("\n" + "="*80)
print("COMPARISON: Individual Models vs Ensemble")
print("="*80)

print("\n🔷 GNN Only:")
evaluate_predictions(test_y, test_gnn, "GNN")

print("\n🔶 XGBoost Only:")
evaluate_predictions(test_y, test_xgb, "XGBoost")

print("\n🔷 LightGBM Only:")
evaluate_predictions(test_y, test_lgb, "LightGBM")

print("\n🌟 STACKING ENSEMBLE:")
acc1, acc2, acc5 = evaluate_predictions(test_y, test_ensemble, "Ensemble")

# 6. Save ensemble
print("\n💾 Saving ensemble...")
ensemble_data = {
    'meta_model': meta_model,
    'gnn_scalers': gnn_scalers,
    'xgb_scaler': xgb_scaler,
    'lgb_scaler': lgb_scaler
}

with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_data, f)

print("\n" + "="*80)
print("✅ ENSEMBLE COMPLETE!")
print("="*80)
print(f"\n🎯 Final Ensemble Performance:")
print(f"  ±1 min: {acc1:.1f}%")
print(f"  ±2 min: {acc2:.1f}%")
print(f"  ±5 min: {acc5:.1f}%")
