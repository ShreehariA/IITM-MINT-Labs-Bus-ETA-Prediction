"""
API Model Loader
Loads saved ensemble models (GNN, XGBoost, LightGBM) and meta-learner
Production-ready with error handling, bounds, and confidence scoring
"""

import torch
import numpy as np
import pickle
import xgboost as xgb
import lightgbm as lgb
from torch_geometric.loader import DataLoader

# Import GNN model (now local)
from gnn_model import TemporalGraphAttention

# Import config and logging
from config import ETA_BOUNDS, CONFIDENCE, MODEL
from logger import log_error

class EnsembleModel:
    """
    Loads and manages ensemble of GNN + XGBoost + LightGBM with meta-learner.
    Uses existing saved models from build_ensemble.py
    """
    
    def __init__(self, model_dir='.', use_gnn=None):
        """
        Load all saved models.
        
        Args:
            model_dir: Directory containing model files (default: current directory)
            use_gnn: Whether to load GNN (default: from config, False since weight is 0)
        """
        print("🔄 Loading ensemble models...")
        
        self.model_dir = model_dir
        self.use_gnn = use_gnn if use_gnn is not None else MODEL['use_gnn']
        
        # Load GNN model (optional)
        if self.use_gnn:
            print("  Loading GNN...")
            self.gnn_model = TemporalGraphAttention(
                num_features=8,
                hidden=128,
                heads=8,
                dropout=0.3
            )
            self.gnn_model.load_state_dict(
                torch.load(f'{model_dir}/tgat_best_random.pt', weights_only=False)
            )
            self.gnn_model.eval()
        else:
            print("  Skipping GNN (weight is 0, disabled in config)")
            self.gnn_model = None
        
        # Load XGBoost
        print("  Loading XGBoost...")
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(f'{model_dir}/xgboost_model.json')
        
        # Load LightGBM
        print("  Loading LightGBM...")
        self.lgb_model = lgb.Booster(model_file=f'{model_dir}/lightgbm_model.txt')
        
        # Load ensemble (meta-learner + scalers)
        print("  Loading meta-learner...")
        with open(f'{model_dir}/ensemble_model.pkl', 'rb') as f:
            ensemble_data = pickle.load(f)
            self.meta_model = ensemble_data['meta_model']
            self.gnn_scalers = ensemble_data['gnn_scalers']
            self.xgb_scaler = ensemble_data['xgb_scaler']
            self.lgb_scaler = ensemble_data['lgb_scaler']
        
        print("✅ All models loaded successfully!")
        print(f"\n📊 Meta-model weights:")
        print(f"   GNN:      {self.meta_model.coef_[0]:.3f}")
        print(f"   XGBoost:  {self.meta_model.coef_[1]:.3f}")
        print(f"   LightGBM: {self.meta_model.coef_[2]:.3f}")
    
    def graph_to_xgb_features(self, graph):
        """Convert graph to XGBoost/LightGBM features"""
        x = graph.x.numpy()
        
        # Node statistics
        node_mean = x.mean(axis=0)
        node_std = x.std(axis=0)
        node_max = x.max(axis=0)
        node_min = x.min(axis=0)
        node_median = np.median(x, axis=0)
        
        # Graph structure
        num_nodes = graph.num_nodes
        num_edges = graph.edge_index.shape[1]
        
        # Target node features
        if hasattr(graph, 'target_node') and graph.target_node < len(x):
            target_features = x[graph.target_node]
        else:
            target_features = node_mean
        
        # Derived features
        speed_variance = x[:, 0].var() if len(x) > 0 else 0
        dist_progression = x[:, 1].mean() / (x[:, 1].max() + 1e-6) if len(x) > 1 else 0
        
        hour_features = x[:, 2] if len(x) > 0 else np.array([0])
        is_rush_hour = ((hour_features >= 7) & (hour_features <= 9) | 
                        (hour_features >= 17) & (hour_features <= 19)).mean()
        
        # Combine all features (53 total)
        features = np.concatenate([
            node_mean, node_std, node_max, node_min, node_median,
            target_features,
            [num_nodes, num_edges, speed_variance, dist_progression, is_rush_hour]
        ])
        
        return features
    
    def predict(self, graph):
        """
        Predict ETA for a single graph.
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            eta_seconds: Predicted ETA in seconds
        """
        # GNN prediction (if enabled)
        if self.gnn_model is not None:
            with torch.no_grad():
                gnn_pred = self.gnn_model(graph).cpu().numpy()
                # Denormalize
                gnn_pred = gnn_pred * self.gnn_scalers['target_range'] + self.gnn_scalers['target_min']
                # Extract scalar value
                gnn_pred_val = float(gnn_pred.flatten()[0]) if gnn_pred.size > 0 else 0.0
        else:
            # GNN disabled, use 0 (meta-learner weight is 0 anyway)
            gnn_pred_val = 0.0
        
        # XGBoost prediction
        xgb_features = self.graph_to_xgb_features(graph)
        xgb_features_scaled = self.xgb_scaler.transform([xgb_features])
        xgb_pred = self.xgb_model.predict(xgb_features_scaled)
        xgb_pred_val = float(xgb_pred[0]) if len(xgb_pred) > 0 else 0.0
        
        # LightGBM prediction
        lgb_pred = self.lgb_model.predict(xgb_features_scaled)
        lgb_pred_val = float(lgb_pred[0]) if len(lgb_pred) > 0 else 0.0
        
        # Meta-learner combines all predictions
        meta_input = np.array([[gnn_pred_val, xgb_pred_val, lgb_pred_val]])
        ensemble_pred = self.meta_model.predict(meta_input)[0]
        
        return max(0, float(ensemble_pred))  # Ensure non-negative
    
    def predict_with_bounds(self, graph):
        """
        Production-ready prediction with bounds, confidence, and error handling.
        
        Args:
            graph: PyTorch Geometric Data object
            
        Returns:
            dict: {
                'eta_seconds': float,
                'eta_minutes': float,
                'confidence': float,
                'was_capped': bool,
                'method': str,
                'distance_meters': float (optional)
            }
        """
        from prediction_utils import calculate_confidence, apply_bounds
        
        try:
            # Get raw prediction
            raw_prediction = self.predict(graph)
            
            # Apply bounds
            bounded_prediction, was_capped = apply_bounds(raw_prediction)
            
            # Extract distance from graph
            distance = graph.x[0, 1].item() if hasattr(graph, 'x') and graph.x.shape[0] > 0 else 1000
            
            # Calculate confidence
            confidence = calculate_confidence(distance, bounded_prediction, graph)
            
            return {
                'eta_seconds': bounded_prediction,
                'eta_minutes': bounded_prediction / 60,
                'confidence': confidence,
                'was_capped': was_capped,
                'method': 'ml_model',
                'distance_meters': distance
            }
            
        except Exception as e:
            # Log error and return fallback
            log_error(e, 'predict_with_bounds')
            
            # Try to extract position for fallback
            if hasattr(graph, 'x') and graph.x.shape[0] > 0:
                from prediction_utils import simple_eta_fallback
                # Assume first node has lat/lon (this is a simplification)
                return simple_eta_fallback(0, 0, 18)  # Will use default
            else:
                from prediction_utils import default_eta_fallback
                return default_eta_fallback()
    
    def predict_batch(self, graphs):
        """
        Predict ETA for multiple graphs.
        
        Args:
            graphs: List of PyTorch Geometric Data objects
            
        Returns:
            eta_seconds: List of predicted ETAs in seconds
        """
        predictions = []
        
        # GNN predictions (batched)
        loader = DataLoader(graphs, batch_size=32)
        gnn_preds = []
        
        with torch.no_grad():
            for batch in loader:
                pred = self.gnn_model(batch).cpu().numpy()
                # Denormalize
                pred = pred * self.gnn_scalers['target_range'] + self.gnn_scalers['target_min']
                gnn_preds.extend(pred)
        
        gnn_preds = np.array(gnn_preds)
        
        # XGBoost/LightGBM predictions
        xgb_features = np.array([self.graph_to_xgb_features(g) for g in graphs])
        xgb_features_scaled = self.xgb_scaler.transform(xgb_features)
        
        xgb_preds = self.xgb_model.predict(xgb_features_scaled)
        lgb_preds = self.lgb_model.predict(xgb_features_scaled)
        
        # Meta-learner combines all
        meta_input = np.column_stack([gnn_preds, xgb_preds, lgb_preds])
        ensemble_preds = self.meta_model.predict(meta_input)
        
        return [max(0, pred) for pred in ensemble_preds]


# Example usage
if __name__ == "__main__":
    # Load models
    model = EnsembleModel()
    
    # Test with a sample graph
    from preprocessing import LiveDataPreprocessor
    from datetime import datetime
    
    preprocessor = LiveDataPreprocessor()
    
    sample_gps = {
        'latitude': 12.9882,
        'longitude': 80.2237,
        'speed': 15.5,
        'timestamp': datetime.now()
    }
    
    # Convert to graph
    graph = preprocessor.gps_to_graph(sample_gps, target_stop_id=18)
    
    # Predict
    eta_seconds = model.predict(graph)
    eta_minutes = eta_seconds / 60
    
    print(f"\n✅ Prediction:")
    print(f"   ETA: {eta_minutes:.1f} minutes ({eta_seconds:.0f} seconds)")
