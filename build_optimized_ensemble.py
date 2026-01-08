"""
MAXIMUM HEAVY Optimized Ensemble with Hyperparameter Tuning
100 trials, 3 base models (XGBoost, LightGBM, CatBoost)
Multiple meta-learners tested
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

print("="*80)
print("OPTIMIZED ENSEMBLE WITH HYPERPARAMETER TUNING")
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

# Optuna objective for ensemble
def objective(trial):
    # XGBoost hyperparameters (EXPANDED RANGE)
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
    
    # LightGBM hyperparameters (EXPANDED RANGE)
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
    
    # CatBoost hyperparameters (NEW!)
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
    
    # Stack predictions (3 models now!)
    X_train_meta = np.column_stack([xgb_train_pred, lgb_train_pred, cat_train_pred])
    X_val_meta = np.column_stack([xgb_val_pred, lgb_val_pred, cat_val_pred])
    
    # Meta-learner type (EXPANDED OPTIONS)
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
    
    # Calculate ±1 min accuracy
    acc_1min = (np.abs(y_val - val_ensemble_pred) <= 60).mean() * 100
    
    return acc_1min

# Run optimization
print("\n🔍 Running MAXIMUM HEAVY Optuna optimization (100 trials)...")
print("  This will take ~60-90 minutes...")
print("  Optimizing: XGBoost + LightGBM + CatBoost + Meta-learner")
print("  Go grab coffee! ☕")

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\n✅ Best ±1 min accuracy: {study.best_value:.1f}%")
print(f"\n📋 Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final ensemble with best params
print("\n🚀 Training final ensemble with best parameters...")

# Extract params
best_params = study.best_params

xgb_params = {
    'n_estimators': best_params['xgb_n_estimators'],
    'max_depth': best_params['xgb_max_depth'],
    'learning_rate': best_params['xgb_lr'],
    'subsample': best_params['xgb_subsample'],
    'colsample_bytree': best_params['xgb_colsample'],
    'colsample_bylevel': best_params['xgb_colsample_level'],
    'min_child_weight': best_params['xgb_min_child'],
    'gamma': best_params['xgb_gamma'],
    'reg_alpha': best_params['xgb_alpha'],
    'reg_lambda': best_params['xgb_lambda'],
    'max_delta_step': best_params['xgb_max_delta'],
    'random_state': 42,
    'n_jobs': -1
}

lgb_params = {
    'n_estimators': best_params['lgb_n_estimators'],
    'num_leaves': best_params['lgb_num_leaves'],
    'learning_rate': best_params['lgb_lr'],
    'feature_fraction': best_params['lgb_feature_fraction'],
    'bagging_fraction': best_params['lgb_bagging_fraction'],
    'bagging_freq': best_params['lgb_bagging_freq'],
    'max_depth': best_params['lgb_max_depth'],
    'min_child_samples': best_params['lgb_min_child'],
    'min_child_weight': best_params['lgb_min_child_weight'],
    'reg_alpha': best_params['lgb_alpha'],
    'reg_lambda': best_params['lgb_lambda'],
    'min_split_gain': best_params['lgb_min_split_gain'],
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

cat_params = {
    'iterations': best_params['cat_iterations'],
    'depth': best_params['cat_depth'],
    'learning_rate': best_params['cat_lr'],
    'l2_leaf_reg': best_params['cat_l2'],
    'border_count': best_params['cat_border'],
    'bagging_temperature': best_params['cat_bagging_temp'],
    'random_strength': best_params['cat_random_strength'],
    'random_state': 42,
    'verbose': False
}

# Train models
xgb_final = xgb.XGBRegressor(**xgb_params)
xgb_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

lgb_final = lgb.LGBMRegressor(**lgb_params)
lgb_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

cat_final = CatBoostRegressor(**cat_params)
cat_final.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

# Meta-learner with 3 base models
xgb_train_pred = xgb_final.predict(X_train)
lgb_train_pred = lgb_final.predict(X_train)
cat_train_pred = cat_final.predict(X_train)
X_train_meta = np.column_stack([xgb_train_pred, lgb_train_pred, cat_train_pred])

meta_type = best_params['meta_type']
if meta_type == 'ridge':
    meta_final = Ridge(alpha=best_params['ridge_alpha'])
elif meta_type == 'lasso':
    meta_final = Lasso(alpha=best_params['lasso_alpha'], max_iter=10000)
elif meta_type == 'elasticnet':
    meta_final = ElasticNet(alpha=best_params['elastic_alpha'], l1_ratio=best_params['elastic_l1'], max_iter=10000)
elif meta_type == 'gbm':
    meta_final = GradientBoostingRegressor(
        n_estimators=best_params['gbm_n_estimators'],
        max_depth=best_params['gbm_max_depth'],
        learning_rate=best_params['gbm_lr'],
        subsample=best_params['gbm_subsample'],
        random_state=42
    )
else:  # xgb_meta
    meta_final = xgb.XGBRegressor(
        n_estimators=best_params['meta_xgb_n_estimators'],
        max_depth=best_params['meta_xgb_max_depth'],
        learning_rate=best_params['meta_xgb_lr'],
        random_state=42,
        n_jobs=-1
    )

meta_final.fit(X_train_meta, y_train)

# Evaluate on test
xgb_test_pred = xgb_final.predict(X_test)
lgb_test_pred = lgb_final.predict(X_test)
cat_test_pred = cat_final.predict(X_test)
X_test_meta = np.column_stack([xgb_test_pred, lgb_test_pred, cat_test_pred])

test_ensemble_pred = meta_final.predict(X_test_meta)

mae = np.mean(np.abs(y_test - test_ensemble_pred)) / 60
rmse = np.sqrt(np.mean((y_test - test_ensemble_pred)**2)) / 60
acc_1min = (np.abs(y_test - test_ensemble_pred) <= 60).mean() * 100
acc_2min = (np.abs(y_test - test_ensemble_pred) <= 120).mean() * 100
acc_5min = (np.abs(y_test - test_ensemble_pred) <= 300).mean() * 100

print(f"\n📈 Optimized Ensemble Test Results:")
print(f"  MAE:  {mae:.2f} minutes")
print(f"  RMSE: {rmse:.2f} minutes")
print(f"  ±1 min: {acc_1min:.1f}%")
print(f"  ±2 min: {acc_2min:.1f}%")
print(f"  ±5 min: {acc_5min:.1f}%")

# Save models
xgb_final.save_model('xgboost_optimized_ensemble.json')
lgb_final.booster_.save_model('lightgbm_optimized_ensemble.txt')
cat_final.save_model('catboost_optimized_ensemble.cbm')

with open('optimized_ensemble.pkl', 'wb') as f:
    pickle.dump({
        'meta_model': meta_final,
        'scaler': scaler,
        'best_params': best_params
    }, f)

print("\n💾 Models saved!")
print("\n✅ OPTIMIZED ENSEMBLE COMPLETE!")
print("="*80)

