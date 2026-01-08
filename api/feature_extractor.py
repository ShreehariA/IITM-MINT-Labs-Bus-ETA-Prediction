"""
API Preprocessing Module
Converts live GPS data to graph features using existing preprocessing_unified approach
"""

import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch_geometric.data import Data

# Import utilities (now local)
from utils import (
    haversine_distance,
    find_nearest_stop,
    identify_route_from_position
)

# Import stop metadata
from stops import STOPS

# Import config and logging
from config import GPS_VALIDATION
from logger import log_validation_failure

def validate_gps_data(gps_data):
    """
    Validate GPS data before processing.
    
    Args:
        gps_data: Dict with latitude, longitude, speed, timestamp
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If validation fails
    """
    lat = gps_data.get('latitude', 0)
    lon = gps_data.get('longitude', 0)
    speed = gps_data.get('speed', 0)
    
    # Check for invalid (0,0) coordinates
    if lat == 0 and lon == 0:
        log_validation_failure('coordinates', (lat, lon), 'Invalid (0,0) coordinates')
        raise ValueError("Invalid GPS: (0,0) coordinates detected")
    
    # Check latitude bounds
    if not (GPS_VALIDATION['lat_min'] <= lat <= GPS_VALIDATION['lat_max']):
        log_validation_failure('latitude', lat, f"Outside range [{GPS_VALIDATION['lat_min']}, {GPS_VALIDATION['lat_max']}]")
        raise ValueError(f"Invalid latitude: {lat}")
    
    # Check longitude bounds
    if not (GPS_VALIDATION['lon_min'] <= lon <= GPS_VALIDATION['lon_max']):
        log_validation_failure('longitude', lon, f"Outside range [{GPS_VALIDATION['lon_min']}, {GPS_VALIDATION['lon_max']}]")
        raise ValueError(f"Invalid longitude: {lon}")
    
    # Check Chennai/IIT Madras area bounds
    if not (GPS_VALIDATION['chennai_lat_min'] <= lat <= GPS_VALIDATION['chennai_lat_max']):
        log_validation_failure('latitude', lat, 'Outside Chennai area')
        raise ValueError(f"GPS outside Chennai area: latitude {lat}")
    
    if not (GPS_VALIDATION['chennai_lon_min'] <= lon <= GPS_VALIDATION['chennai_lon_max']):
        log_validation_failure('longitude', lon, 'Outside Chennai area')
        raise ValueError(f"GPS outside Chennai area: longitude {lon}")
    
    # Check speed bounds
    if not (GPS_VALIDATION['speed_min'] <= speed <= GPS_VALIDATION['speed_max']):
        log_validation_failure('speed', speed, f"Outside range [{GPS_VALIDATION['speed_min']}, {GPS_VALIDATION['speed_max']}]")
        raise ValueError(f"Invalid speed: {speed} km/h")
    
    return True

# Import graph builder
from graph import BusGraphBuilder

class LiveDataPreprocessor:
    """
    Converts live GPS data to graph format for model prediction.
    Uses existing graph_builder.py approach.
    """
    
    def __init__(self):
        """Initialize preprocessor (graph builder created on-demand)"""
        # Don't initialize graph_builder here since it requires a dataframe
        # We'll create graphs directly in gps_to_graph method
        pass
    
    def gps_to_graph(self, gps_data: dict, target_stop_id: int, route_id: str = None) -> Data:
        """
        Convert single GPS point to graph for prediction.
        
        Args:
            gps_data: {
                'latitude': float,
                'longitude': float,
                'speed': float,
                'timestamp': datetime or str
            }
            target_stop_id: Target stop ID (1-29)
            route_id: Optional route ID (e.g., 'HOSTEL_MAIN')
            
        Returns:
            PyTorch Geometric Data object (graph)
        """
        # Parse timestamp
        if isinstance(gps_data['timestamp'], str):
            timestamp = pd.to_datetime(gps_data['timestamp'])
        else:
            timestamp = gps_data['timestamp']
        
        # Find nearest stop if not provided
        if route_id is None:
            route_id = identify_route_from_position(
                gps_data['latitude'],
                gps_data['longitude']
            )
        
        # Get target stop info
        target_stop = STOPS[target_stop_id]
        
        # Calculate distance to target
        dist_to_target = haversine_distance(
            gps_data['latitude'], gps_data['longitude'],
            target_stop['lat'], target_stop['lon']
        )
        
        # Extract temporal features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        is_rush_hour = 1.0 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0.0
        
        # Create node features (8 features per node)
        # For real-time: Single node representing current position
        node_features = np.array([[
            gps_data['speed'],           # 0: speed
            dist_to_target,              # 1: distance to target
            hour,                        # 2: hour of day
            day_of_week,                 # 3: day of week
            gps_data['latitude'],        # 4: latitude
            gps_data['longitude'],       # 5: longitude
            target_stop['lat'],          # 6: target latitude
            target_stop['lon']           # 7: target longitude
        ]], dtype=np.float32)
        
        # Create graph with single node
        # No edges for single node
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        graph = Data(
            x=torch.FloatTensor(node_features),
            edge_index=edge_index,
            num_nodes=1,
            target_node=0  # Single node is the target
        )
        
        # Add metadata
        graph.route_id = route_id
        graph.target_stop_id = target_stop_id
        graph.timestamp = timestamp
        
        return graph
    
    def batch_gps_to_graphs(self, gps_data_list: list, target_stop_id: int) -> list:
        """
        Convert multiple GPS points to graphs.
        
        Args:
            gps_data_list: List of GPS data dicts
            target_stop_id: Target stop ID
            
        Returns:
            List of PyTorch Geometric Data objects
        """
        graphs = []
        for gps_data in gps_data_list:
            try:
                graph = self.gps_to_graph(gps_data, target_stop_id)
                graphs.append(graph)
            except Exception as e:
                print(f"⚠️  Error processing GPS data: {e}")
                continue
        
        return graphs


# Example usage
if __name__ == "__main__":
    # Test with sample GPS data
    preprocessor = LiveDataPreprocessor()
    
    sample_gps = {
        'latitude': 12.9882,
        'longitude': 80.2237,
        'speed': 15.5,
        'timestamp': datetime.now()
    }
    
    # Convert to graph
    graph = preprocessor.gps_to_graph(sample_gps, target_stop_id=18)  # Main Gate
    
    print(f"\n✅ Graph created:")
    print(f"   Nodes: {graph.num_nodes}")
    print(f"   Features shape: {graph.x.shape}")
    print(f"   Target stop: {graph.target_stop_id}")
    print(f"   Route: {graph.route_id}")
