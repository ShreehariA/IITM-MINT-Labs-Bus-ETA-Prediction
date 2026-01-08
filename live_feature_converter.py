"""
Live Data to Model Features Converter
Converts real-time API data to 53-feature format expected by ensemble model
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import math

class LiveFeatureConverter:
    """
    Converts live GPS data to the 53-feature format used in training.
    
    Training features were:
    - Node statistics (mean, std, max, min, median) × 8 base features = 40
    - Target node features × 8 = 8
    - Graph structure (num_nodes, num_edges) = 2
    - Derived features (speed_variance, dist_progression, is_rush_hour) = 3
    Total: 53 features
    """
    
    def __init__(self, stop_metadata: Dict):
        """
        Args:
            stop_metadata: Dict mapping stop_id to {name, lat, lon, routes}
        """
        self.stops = stop_metadata
        
    def convert(self, vehicle_data: Dict, target_stop_id: int) -> np.ndarray:
        """
        Convert live vehicle data to 53-feature vector.
        
        Args:
            vehicle_data: {
                'latitude': float,
                'longitude': float,
                'speed': float,
                'heading': float,
                'vehicleno': str,
                'timestamp': str (ISO format)
            }
            target_stop_id: ID of the stop to predict ETA for
            
        Returns:
            np.ndarray of shape (53,)
        """
        target_stop = self.stops[target_stop_id]
        
        # Extract base features from current vehicle state
        base_features = self._extract_base_features(vehicle_data, target_stop)
        
        # Since we only have ONE vehicle position (not a graph),
        # we approximate graph statistics
        node_mean = base_features  # Mean = current value
        node_std = np.zeros(8)     # No variance with single point
        node_max = base_features    # Max = current value
        node_min = base_features    # Min = current value
        node_median = base_features # Median = current value
        
        # Target node features (same as base for single vehicle)
        target_features = base_features
        
        # Graph structure (single vehicle = 1 node, 0 edges)
        num_nodes = 1
        num_edges = 0
        
        # Derived features
        speed_variance = 0  # No variance with single point
        dist_progression = self._calculate_distance_progression(
            vehicle_data, target_stop
        )
        is_rush_hour = self._is_rush_hour(vehicle_data['timestamp'])
        
        # Combine all features (53 total)
        features = np.concatenate([
            node_mean,           # 8 features
            node_std,            # 8 features
            node_max,            # 8 features
            node_min,            # 8 features
            node_median,         # 8 features
            target_features,     # 8 features
            [num_nodes],         # 1 feature
            [num_edges],         # 1 feature
            [speed_variance],    # 1 feature
            [dist_progression],  # 1 feature
            [is_rush_hour]       # 1 feature
        ])
        
        assert len(features) == 53, f"Expected 53 features, got {len(features)}"
        return features
    
    def _extract_base_features(self, vehicle: Dict, target_stop: Dict) -> np.ndarray:
        """
        Extract 8 base features from vehicle and target stop.
        
        These match the original graph node features:
        0: speed
        1: distance_to_target
        2: hour_of_day
        3: day_of_week
        4: latitude
        5: longitude
        6: target_latitude
        7: target_longitude
        """
        # Calculate distance to target
        distance = self._haversine_distance(
            vehicle['latitude'], vehicle['longitude'],
            target_stop['latitude'], target_stop['longitude']
        )
        
        # Extract time features
        dt = datetime.fromisoformat(vehicle['timestamp'].replace('Z', '+00:00'))
        hour = dt.hour
        day_of_week = dt.weekday()
        
        return np.array([
            vehicle['speed'],
            distance,
            hour,
            day_of_week,
            vehicle['latitude'],
            vehicle['longitude'],
            target_stop['latitude'],
            target_stop['longitude']
        ])
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points in meters using Haversine formula.
        """
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_phi/2)**2 + 
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _calculate_distance_progression(self, vehicle: Dict, 
                                       target_stop: Dict) -> float:
        """
        Calculate route progression (0 to 1).
        
        For now, use simple distance-based heuristic.
        Later can improve with actual route path.
        """
        distance = self._haversine_distance(
            vehicle['latitude'], vehicle['longitude'],
            target_stop['latitude'], target_stop['longitude']
        )
        
        # Normalize by typical max distance on campus (~5km)
        max_distance = 5000
        progression = min(distance / max_distance, 1.0)
        
        return progression
    
    def _is_rush_hour(self, timestamp: str) -> float:
        """
        Check if current time is rush hour.
        Rush hours: 7-9 AM, 5-7 PM
        
        Returns 1.0 if rush hour, 0.0 otherwise.
        """
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        hour = dt.hour
        
        is_rush = (7 <= hour <= 9) or (17 <= hour <= 19)
        return 1.0 if is_rush else 0.0


# Example usage
if __name__ == "__main__":
    # Example stop metadata (you'll need to populate this)
    stop_metadata = {
        1: {
            'name': 'Main Gate',
            'latitude': 12.9916,
            'longitude': 80.2336,
            'routes': ['MAIN_HOSTEL', 'ED_RP']
        },
        # Add all 29 stops...
    }
    
    # Example live vehicle data
    vehicle_data = {
        'latitude': 12.9882,
        'longitude': 80.2237,
        'speed': 15.5,
        'heading': 175.3,
        'vehicleno': 'EV08',
        'timestamp': '2026-01-02T06:30:00Z'
    }
    
    # Convert to model features
    converter = LiveFeatureConverter(stop_metadata)
    features = converter.convert(vehicle_data, target_stop_id=1)
    
    print(f"Generated {len(features)} features:")
    print(features)
    
    # Now you can feed this to your ensemble model:
    # prediction = ensemble_model.predict([features])
