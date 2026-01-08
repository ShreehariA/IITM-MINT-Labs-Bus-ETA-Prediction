"""
MAXIMUM HEAVY Ray Tune Multi-Objective Optimization
Parallel hyperparameter search using all CPU cores
Goal: Push all metrics to their maximum potential!
"""
import torch
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RAY TUNE MAXIMUM HEAVY MULTI-OBJECTIVE OPTIMIZATION")
print("Goal: Push ALL metrics to their MAXIMUM potential!")
print("="*80)

# Load and prepare data
print("\n📂 Loading data...")
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

# Extract features
print("🔄 Extracting features...")
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

print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Ray Tune trainable function
def train_ensemble(config):
    """Train ensemble with given config - optimized for Ray Tune"""
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=config['xgb_n_estimators'],
        max_depth=config['xgb_max_depth'],
        learning_rate=config['xgb_lr'],
        subsample=config['xgb_subsample'],
        colsample_bytree=config['xgb_colsample'],
        colsample_bylevel=config['xgb_colsample_level'],
        min_child_weight=config['xgb_min_child'],
        gamma=config['xgb_gamma'],
        reg_alpha=config['xgb_alpha'],
        reg_lambda=config['xgb_lambda'],
        max_delta_step=config['xgb_max_delta'],
        random_state=42,
        n_jobs=1  # Ray handles parallelism
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=config['lgb_n_estimators'],
        num_leaves=config['lgb_num_leaves'],
        learning_rate=config['lgb_lr'],
        feature_fraction=config['lgb_feature_fraction'],
        bagging_fraction=config['lgb_bagging_fraction'],
        bagging_freq=config['lgb_bagging_freq'],
        max_depth=config['lgb_max_depth'],
        min_child_samples=config['lgb_min_child'],
        min_child_weight=config['lgb_min_child_weight'],
        reg_alpha=config['lgb_alpha'],
        reg_lambda=config['lgb_lambda'],
        min_split_gain=config['lgb_min_split_gain'],
        random_state=42,
        n_jobs=1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=config['cat_iterations'],
        depth=config['cat_depth'],
        learning_rate=config['cat_lr'],
        l2_leaf_reg=config['cat_l2'],
        border_count=config['cat_border'],
        bagging_temperature=config['cat_bagging_temp'],
        random_strength=config['cat_random_strength'],
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), 
                  early_stopping_rounds=50, verbose=False)
    
    # Get predictions
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_val_pred = xgb_model.predict(X_val)
    
    lgb_train_pred = lgb_model.predict(X_train)
    lgb_val_pred = lgb_model.predict(X_val)
    
    cat_train_pred = cat_model.predict(X_train)
    cat_val_pred = cat_model.predict(X_val)
    
    # Stack predictions
    X_train_meta = np.column_stack([xgb_train_pred, lgb_train_pred, cat_train_pred])
    X_val_meta = np.column_stack([xgb_val_pred, lgb_val_pred, cat_val_pred])
    
    # Meta-learner
    if config['meta_type'] == 'ridge':
        meta_model = Ridge(alpha=config['ridge_alpha'])
    elif config['meta_type'] == 'lasso':
        meta_model = Lasso(alpha=config['lasso_alpha'], max_iter=10000)
    elif config['meta_type'] == 'elasticnet':
        meta_model = ElasticNet(alpha=config['elastic_alpha'], 
                                l1_ratio=config['elastic_l1'], max_iter=10000)
    elif config['meta_type'] == 'gbm':
        meta_model = GradientBoostingRegressor(
            n_estimators=config['gbm_n_estimators'],
            max_depth=config['gbm_max_depth'],
            learning_rate=config['gbm_lr'],
            subsample=config['gbm_subsample'],
            random_state=42
        )
    else:  # xgb_meta
        meta_model = xgb.XGBRegressor(
            n_estimators=config['meta_xgb_n_estimators'],
            max_depth=config['meta_xgb_max_depth'],
            learning_rate=config['meta_xgb_lr'],
            random_state=42,
            n_jobs=1
        )
    
    meta_model.fit(X_train_meta, y_train)
    
    # Predict on validation
    val_ensemble_pred = meta_model.predict(X_val_meta)
    
    # Calculate ALL three accuracy metrics
    errors = np.abs(y_val - val_ensemble_pred)
    acc_1min = (errors <= 60).mean() * 100
    acc_2min = (errors <= 120).mean() * 100
    acc_5min = (errors <= 300).mean() * 100
    
    # Multi-objective score with balanced weights
    score = (0.3 * acc_1min +   # 30% weight on ±1min
             0.3 * acc_2min +   # 30% weight on ±2min
             0.4 * acc_5min)    # 40% weight on ±5min
    
    return {
        "score": score,
        "acc_1min": acc_1min,
        "acc_2min": acc_2min,
        "acc_5min": acc_5min
    }

# Define search space - MAXIMUM HEAVY
search_space = {
    # XGBoost - 11 hyperparameters
    'xgb_n_estimators': tune.randint(2000, 10000),  # Extended range
    'xgb_max_depth': tune.randint(3, 25),
    'xgb_lr': tune.loguniform(0.0003, 0.15),
    'xgb_subsample': tune.uniform(0.4, 1.0),
    'xgb_colsample': tune.uniform(0.4, 1.0),
    'xgb_colsample_level': tune.uniform(0.4, 1.0),
    'xgb_min_child': tune.randint(1, 25),
    'xgb_gamma': tune.uniform(0, 12),
    'xgb_alpha': tune.uniform(0, 6),
    'xgb_lambda': tune.uniform(0, 6),
    'xgb_max_delta': tune.randint(0, 12),
    
    # LightGBM - 12 hyperparameters
    'lgb_n_estimators': tune.randint(2000, 10000),
    'lgb_num_leaves': tune.randint(15, 350),
    'lgb_lr': tune.loguniform(0.0003, 0.15),
    'lgb_feature_fraction': tune.uniform(0.4, 1.0),
    'lgb_bagging_fraction': tune.uniform(0.4, 1.0),
    'lgb_bagging_freq': tune.randint(1, 12),
    'lgb_max_depth': tune.randint(3, 25),
    'lgb_min_child': tune.randint(3, 120),
    'lgb_min_child_weight': tune.loguniform(1e-6, 12),
    'lgb_alpha': tune.uniform(0, 6),
    'lgb_lambda': tune.uniform(0, 6),
    'lgb_min_split_gain': tune.uniform(0, 6),
    
    # CatBoost - 7 hyperparameters
    'cat_iterations': tune.randint(2000, 10000),
    'cat_depth': tune.randint(3, 14),
    'cat_lr': tune.loguniform(0.0003, 0.15),
    'cat_l2': tune.uniform(0.5, 12),
    'cat_border': tune.randint(28, 280),
    'cat_bagging_temp': tune.uniform(0, 1.2),
    'cat_random_strength': tune.uniform(0, 12),
    
    # Meta-learner - 5 types with expanded params
    'meta_type': tune.choice(['ridge', 'lasso', 'elasticnet', 'gbm', 'xgb_meta']),
    'ridge_alpha': tune.loguniform(0.0005, 150.0),
    'lasso_alpha': tune.loguniform(0.0005, 150.0),
    'elastic_alpha': tune.loguniform(0.0005, 150.0),
    'elastic_l1': tune.uniform(0.0, 1.0),
    'gbm_n_estimators': tune.randint(40, 600),
    'gbm_max_depth': tune.randint(2, 10),
    'gbm_lr': tune.loguniform(0.0005, 0.6),
    'gbm_subsample': tune.uniform(0.4, 1.0),
    'meta_xgb_n_estimators': tune.randint(80, 1200),
    'meta_xgb_max_depth': tune.randint(2, 8),
    'meta_xgb_lr': tune.uniform(0.005, 0.35),
}

# Run Ray Tune optimization
print("\n🚀 Starting Ray Tune MAXIMUM HEAVY optimization...")
print(f"  Trials: 150 (HEAVY search)")
print(f"  Parallel workers: 12 cores (leaving 2 for system)")
print(f"  ASHA scheduler with early stopping")
print(f"  Optuna search algorithm (TPE)")
print(f"  Expected time: ~20-30 minutes (vs 90+ min sequential)")
print(f"  Optimizing: 0.3×(±1min) + 0.3×(±2min) + 0.4×(±5min)")

# Configure ASHA scheduler for aggressive early stopping
scheduler = ASHAScheduler(
    max_t=150,
    grace_period=15,  # Give trials at least 15 iterations
    reduction_factor=3,
    brackets=3
)

# Configure Optuna search
search_alg = OptunaSearch(
    metric="score",
    mode="max"
)

# Run optimization with updated API
import os
tuner = tune.Tuner(
    train_ensemble,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="score",
        mode="max",
        num_samples=150,  # MAXIMUM HEAVY: 150 trials
        scheduler=scheduler,
        search_alg=search_alg,
        max_concurrent_trials=12,  # Use 12 cores
    ),
    run_config=air.RunConfig(
        name="multi_objective_ensemble",
        storage_path=os.path.abspath("./ray_results"),  # Absolute path required
        verbose=1,
        failure_config=air.FailureConfig(fail_fast=False)
    )
)

# Run and get results
results = tuner.fit()

# Get best result
best_result = results.get_best_result(metric="score", mode="max")
best_config = best_result.config
best_metrics = best_result.metrics

print("\n" + "="*80)
print("BEST CONFIGURATION FOUND")
print("="*80)

print(f"\n✅ Best multi-objective score: {best_metrics['score']:.2f}")
print(f"\n📊 Best trial metrics:")
print(f"  ±1 min: {best_metrics['acc_1min']:.1f}%")
print(f"  ±2 min: {best_metrics['acc_2min']:.1f}%")
print(f"  ±5 min: {best_metrics['acc_5min']:.1f}%")

# Analyze top 10 trials
print("\n📋 Top 10 trials:")
print(f"{'Rank':<6} {'±1min':<8} {'±2min':<8} {'±5min':<8} {'Score':<8}")
print("-" * 42)

df = results.get_dataframe()
df_sorted = df.sort_values('score', ascending=False).head(10)
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    if 'acc_1min' in row and 'acc_2min' in row and 'acc_5min' in row:
        print(f"{i+1:<6} {row['acc_1min']:<8.1f} {row['acc_2min']:<8.1f} "
              f"{row['acc_5min']:<8.1f} {row['score']:<8.2f}")

# Save results
print("\n💾 Saving results...")
with open('ray_tune_multi_objective_results.pkl', 'wb') as f:
    pickle.dump({
        'best_config': best_config,
        'best_metrics': best_metrics,
        'results_df': df_sorted
    }, f)

print(f"\n📋 Best configuration:")
for key, value in best_config.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

print("\n✅ RAY TUNE MAXIMUM HEAVY OPTIMIZATION COMPLETE!")
print("="*80)

