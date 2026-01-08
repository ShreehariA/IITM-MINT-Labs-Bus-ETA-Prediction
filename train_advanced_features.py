"""
Advanced feature engineering to push ±1 min accuracy higher
Adds fine-grained temporal and spatial features
"""
import torch
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pickle

print("="*80)
print("ADVANCED FEATURE ENGINEERING FOR 80% TARGET")
print("="*80)

# Load graphs
graphs = torch.load('bus_graphs_full.pt', weights_only=False)

def extract_advanced_features(graph):
    """Extract 100+ advanced features"""
    x = graph.x.numpy()
    
    # Basic stats (40 features)
    node_mean = x.mean(axis=0)
    node_std = x.std(axis=0)
    node_max = x.max(axis=0)
    node_min = x.min(axis=0)
    node_median = np.median(x, axis=0)
    
    # Advanced temporal features (20 features)
    if len(x) > 1:
        # Speed dynamics
        speed_trend = np.polyfit(range(len(x)), x[:, 0], 1)[0] if len(x) > 1 else 0
        speed_acceleration = np.diff(x[:, 0]).mean() if len(x) > 1 else 0
        speed_jerk = np.diff(np.diff(x[:, 0])).mean() if len(x) > 2 else 0
        
        # Distance progression
        dist_trend = np.polyfit(range(len(x)), x[:, 1], 1)[0] if len(x) > 1 else 0
        dist_variance = x[:, 1].var()
        
        # Time patterns
        hour_variance = x[:, 2].var()
        hour_range = x[:, 2].max() - x[:, 2].min()
        
        # Peak hour analysis
        peak_ratio = ((x[:, 3] == 1).sum() / len(x)) if len(x) > 0 else 0
        weekend_ratio = ((x[:, 4] == 1).sum() / len(x)) if len(x) > 0 else 0
        
        # Moving average dynamics
        ma_trend = np.polyfit(range(len(x)), x[:, 5], 1)[0] if len(x) > 1 else 0
        ma_variance = x[:, 5].var()
        
        # Location clustering
        lat_variance = x[:, 6].var()
        lon_variance = x[:, 7].var()
        location_spread = np.sqrt(lat_variance + lon_variance)
        
        # Quartile features
        speed_q25 = np.percentile(x[:, 0], 25)
        speed_q75 = np.percentile(x[:, 0], 75)
        speed_iqr = speed_q75 - speed_q25
        
        dist_q25 = np.percentile(x[:, 1], 25)
        dist_q75 = np.percentile(x[:, 1], 75)
        dist_iqr = dist_q75 - dist_q25
        
        # Skewness and kurtosis approximations
        speed_skew = ((x[:, 0] - node_mean[0])**3).mean() / (node_std[0]**3 + 1e-6)
        dist_skew = ((x[:, 1] - node_mean[1])**3).mean() / (node_std[1]**3 + 1e-6)
    else:
        speed_trend = speed_acceleration = speed_jerk = 0
        dist_trend = dist_variance = 0
        hour_variance = hour_range = 0
        peak_ratio = weekend_ratio = 0
        ma_trend = ma_variance = 0
        lat_variance = lon_variance = location_spread = 0
        speed_q25 = speed_q75 = speed_iqr = 0
        dist_q25 = dist_q75 = dist_iqr = 0
        speed_skew = dist_skew = 0
    
    # Graph structure features (10 features)
    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.shape[1]
    edge_density = num_edges / (num_nodes * (num_nodes - 1) + 1e-6)
    avg_degree = (2 * num_edges) / (num_nodes + 1e-6)
    
    # Target node features (8 features)
    if hasattr(graph, 'target_node') and graph.target_node < len(x):
        target_features = x[graph.target_node]
        target_position = graph.target_node / (num_nodes + 1e-6)
    else:
        target_features = node_mean
        target_position = 0.5
    
    # Segment analysis (15 features)
    if len(x) > 3:
        # First third
        first_third = x[:len(x)//3]
        first_speed = first_third[:, 0].mean()
        first_dist = first_third[:, 1].mean()
        
        # Middle third
        middle_third = x[len(x)//3:2*len(x)//3]
        middle_speed = middle_third[:, 0].mean()
        middle_dist = middle_third[:, 1].mean()
        
        # Last third
        last_third = x[2*len(x)//3:]
        last_speed = last_third[:, 0].mean()
        last_dist = last_third[:, 1].mean()
        
        # Speed changes between segments
        speed_change_1_2 = middle_speed - first_speed
        speed_change_2_3 = last_speed - middle_speed
        
        # Distance progression
        dist_change_1_2 = middle_dist - first_dist
        dist_change_2_3 = last_dist - middle_dist
        
        # Consistency
        speed_consistency = 1 / (1 + abs(speed_change_1_2) + abs(speed_change_2_3))
        dist_consistency = 1 / (1 + abs(dist_change_1_2) + abs(dist_change_2_3))
    else:
        first_speed = middle_speed = last_speed = node_mean[0]
        first_dist = middle_dist = last_dist = node_mean[1]
        speed_change_1_2 = speed_change_2_3 = 0
        dist_change_1_2 = dist_change_2_3 = 0
        speed_consistency = dist_consistency = 1.0
    
    # Interaction features (10 features)
    speed_dist_ratio = node_mean[0] / (node_mean[1] + 1e-6)
    speed_time_interaction = node_mean[0] * node_mean[2]
    dist_time_interaction = node_mean[1] * node_mean[2]
    peak_speed_interaction = node_mean[0] * peak_ratio
    weekend_speed_interaction = node_mean[0] * weekend_ratio
    
    # Efficiency metrics (5 features)
    speed_efficiency = node_mean[0] / (node_max[0] + 1e-6)
    dist_efficiency = node_min[1] / (node_max[1] + 1e-6)
    time_efficiency = num_nodes / (node_mean[2] + 1e-6)
    overall_efficiency = (speed_efficiency + dist_efficiency + time_efficiency) / 3
    
    # Combine all features (108 total)
    features = np.concatenate([
        node_mean,  # 8
        node_std,   # 8
        node_max,   # 8
        node_min,   # 8
        node_median,  # 8
        [speed_trend, speed_acceleration, speed_jerk, dist_trend, dist_variance,
         hour_variance, hour_range, peak_ratio, weekend_ratio, ma_trend,
         ma_variance, lat_variance, lon_variance, location_spread,
         speed_q25, speed_q75, speed_iqr, dist_q25, dist_q75, dist_iqr,
         speed_skew, dist_skew],  # 22
        [num_nodes, num_edges, edge_density, avg_degree],  # 4
        target_features,  # 8
        [target_position],  # 1
        [first_speed, first_dist, middle_speed, middle_dist, last_speed, last_dist,
         speed_change_1_2, speed_change_2_3, dist_change_1_2, dist_change_2_3,
         speed_consistency, dist_consistency],  # 12
        [speed_dist_ratio, speed_time_interaction, dist_time_interaction,
         peak_speed_interaction, weekend_speed_interaction],  # 5
        [speed_efficiency, dist_efficiency, time_efficiency, overall_efficiency]  # 4
    ])
    
    return features

# Extract features
print("\n🔄 Extracting advanced features...")
X_all = np.array([extract_advanced_features(g) for g in graphs])
y_all = np.array([g.y.item() for g in graphs])

print(f"  Feature matrix: {X_all.shape}")
print(f"  Total features: {X_all.shape[1]}")

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

# Train XGBoost with more features
print("\n🚀 Training XGBoost with advanced features...")
xgb_advanced = xgb.XGBRegressor(
    n_estimators=3000,
    max_depth=12,
    learning_rate=0.005,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

xgb_advanced.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# Evaluate
test_pred = xgb_advanced.predict(X_test)

mae = np.mean(np.abs(y_test - test_pred)) / 60
rmse = np.sqrt(np.mean((y_test - test_pred)**2)) / 60
acc_1min = (np.abs(y_test - test_pred) <= 60).mean() * 100
acc_2min = (np.abs(y_test - test_pred) <= 120).mean() * 100
acc_5min = (np.abs(y_test - test_pred) <= 300).mean() * 100

print(f"\n📈 Advanced XGBoost Results:")
print(f"  MAE:  {mae:.2f} minutes")
print(f"  RMSE: {rmse:.2f} minutes")
print(f"  ±1 min: {acc_1min:.1f}%")
print(f"  ±2 min: {acc_2min:.1f}%")
print(f"  ±5 min: {acc_5min:.1f}%")

# Save
xgb_advanced.save_model('xgboost_advanced.json')
with open('advanced_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ ADVANCED MODEL COMPLETE!")
print("="*80)
