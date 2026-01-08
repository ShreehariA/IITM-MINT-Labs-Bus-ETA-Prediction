"""
Multi-Objective Ensemble Optimization
Optimizes for ALL three accuracy thresholds simultaneously
Target: ±5min ~90%, ±2min ~80%, ±1min ~70%
"""
import torch
import numpy as np
import pickle
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MULTI-OBJECTIVE ENSEMBLE OPTIMIZATION")
print("Target: ±5min ~90%, ±2min ~80%, ±1min ~70%")
print("="*80)

# Load graphs
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
print("\n🔄 Extracting features...")
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

# Multi-objective optimization function
def objective(trial):
    # XGBoost hyperparameters
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 2000, 8000),
        'max_depth': trial.suggest_int('xgb_max_depth', 4, 20),
        'learning_rate': trial.suggest_float('xgb_lr', 0.0005, 0.1, log=True),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('xgb_colsample_level', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('xgb_min_child', 1, 20),
        'gamma': trial.suggest_float('xgb_gamma', 0, 10),
        'reg_alpha': trial.suggest_float('xgb_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('xgb_lambda', 0, 5),
        'max_delta_step': trial.suggest_int('xgb_max_delta', 0, 10),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # LightGBM hyperparameters
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 2000, 8000),
        'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('lgb_lr', 0.0005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
        'max_depth': trial.suggest_int('lgb_max_depth', 4, 20),
        'min_child_samples': trial.suggest_int('lgb_min_child', 5, 100),
        'min_child_weight': trial.suggest_float('lgb_min_child_weight', 1e-5, 10),
        'reg_alpha': trial.suggest_float('lgb_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('lgb_lambda', 0, 5),
        'min_split_gain': trial.suggest_float('lgb_min_split_gain', 0, 5),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # CatBoost hyperparameters
    cat_params = {
        'iterations': trial.suggest_int('cat_iterations', 2000, 8000),
        'depth': trial.suggest_int('cat_depth', 4, 12),
        'learning_rate': trial.suggest_float('cat_lr', 0.0005, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('cat_l2', 1, 10),
        'border_count': trial.suggest_int('cat_border', 32, 255),
        'bagging_temperature': trial.suggest_float('cat_bagging_temp', 0, 1),
        'random_strength': trial.suggest_float('cat_random_strength', 0, 10),
        'random_state': 42,
        'verbose': False
    }
    
    # Train base models
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    
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
    
    # Meta-learner type
    meta_type = trial.suggest_categorical('meta_type', ['ridge', 'lasso', 'elasticnet', 'gbm', 'xgb_meta'])
    
    if meta_type == 'ridge':
        alpha = trial.suggest_float('ridge_alpha', 0.001, 100.0, log=True)
        meta_model = Ridge(alpha=alpha)
    elif meta_type == 'lasso':
        alpha = trial.suggest_float('lasso_alpha', 0.001, 100.0, log=True)
        meta_model = Lasso(alpha=alpha, max_iter=10000)
    elif meta_type == 'elasticnet':
        alpha = trial.suggest_float('elastic_alpha', 0.001, 100.0, log=True)
        l1_ratio = trial.suggest_float('elastic_l1', 0.0, 1.0)
        meta_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    elif meta_type == 'gbm':
        meta_model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int('gbm_n_estimators', 50, 500),
            max_depth=trial.suggest_int('gbm_max_depth', 2, 8),
            learning_rate=trial.suggest_float('gbm_lr', 0.001, 0.5, log=True),
            subsample=trial.suggest_float('gbm_subsample', 0.5, 1.0),
            random_state=42
        )
    else:  # xgb_meta
        meta_model = xgb.XGBRegressor(
            n_estimators=trial.suggest_int('meta_xgb_n_estimators', 100, 1000),
            max_depth=trial.suggest_int('meta_xgb_max_depth', 2, 6),
            learning_rate=trial.suggest_float('meta_xgb_lr', 0.01, 0.3),
            random_state=42,
            n_jobs=-1
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
    # Priority: maintain ±5min, improve ±2min, boost ±1min
    score = (0.3 * acc_1min +   # 30% weight on ±1min (improve from 69.7%)
             0.3 * acc_2min +   # 30% weight on ±2min (maintain 84%)
             0.4 * acc_5min)    # 40% weight on ±5min (maintain 93%)
    
    # Store individual metrics for analysis
    trial.set_user_attr('acc_1min', acc_1min)
    trial.set_user_attr('acc_2min', acc_2min)
    trial.set_user_attr('acc_5min', acc_5min)
    
    return score

# Run optimization
print("\n🔍 Running multi-objective optimization (50 trials)...")
print("  Optimizing weighted combination of ±1min, ±2min, ±5min")
print("  Weights: 30% (±1min) + 30% (±2min) + 40% (±5min)")
print("  Expected time: ~30-40 minutes")

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Get best trial
best_trial = study.best_trial
best_params = best_trial.params

print(f"\n✅ Best multi-objective score: {best_trial.value:.2f}")
print(f"\n📊 Best trial metrics:")
print(f"  ±1 min: {best_trial.user_attrs['acc_1min']:.1f}%")
print(f"  ±2 min: {best_trial.user_attrs['acc_2min']:.1f}%")
print(f"  ±5 min: {best_trial.user_attrs['acc_5min']:.1f}%")

# Analyze top trials
print("\n📋 Top 5 trials:")
print(f"{'Rank':<6} {'±1min':<8} {'±2min':<8} {'±5min':<8} {'Score':<8}")
print("-" * 42)
for i, trial in enumerate(sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]):
    print(f"{i+1:<6} {trial.user_attrs['acc_1min']:<8.1f} {trial.user_attrs['acc_2min']:<8.1f} "
          f"{trial.user_attrs['acc_5min']:<8.1f} {trial.value:<8.2f}")

# Save results
with open('multi_objective_results.pkl', 'wb') as f:
    pickle.dump({
        'best_params': best_params,
        'best_trial': best_trial,
        'study': study
    }, f)

print("\n💾 Results saved to multi_objective_results.pkl")
print("\n✅ MULTI-OBJECTIVE OPTIMIZATION COMPLETE!")
print("="*80)
