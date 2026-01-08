"""
Graph Builder for Bus ETA Prediction
Converts GPS trip data into PyTorch Geometric graphs
"""
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from tqdm import tqdm

class BusGraphBuilder:
    """
    Converts bus GPS data into graph structure for GNN
    
    Graph Structure:
    - Nodes: Stops along the route
    - Edges: Sequential connections (Stop1 → Stop2 → Stop3...)
    - Node Features: Speed, distance, time, etc. at each stop
    - Target: ETA to destination stop
    """
    
    def __init__(self, df):
        """
        Initialize graph builder
        
        Args:
            df: Full GPS dataset
        """
        self.df = df
        
        # Get unique stops and create mapping
        self.stops = sorted(df['stop_id'].unique())
        self.stop_to_idx = {stop_id: idx for idx, stop_id in enumerate(self.stops)}
        self.num_stops = len(self.stops)
        
        print(f"✓ Graph Builder initialized")
        print(f"  Stops (nodes): {self.num_stops}")
        print(f"  Stop ID range: {min(self.stops)} to {max(self.stops)}")
    
    def get_stop_sequence(self, trip_id):
        """
        Get ordered sequence of stops for a trip
        
        Returns:
            List of stop_ids in order
        """
        trip_data = self.df[self.df['trip_id'] == trip_id].sort_values('stop_sequence')
        return trip_data['stop_id'].unique().tolist()
    
    def build_edge_index(self, stop_sequence):
        """
        Build edge connections for a sequence of stops
        
        Args:
            stop_sequence: List of stop_ids
            
        Returns:
            edge_index: [2, num_edges] tensor
        """
        edges = []
        for i in range(len(stop_sequence) - 1):
            src_idx = self.stop_to_idx[stop_sequence[i]]
            dst_idx = self.stop_to_idx[stop_sequence[i+1]]
            edges.append([src_idx, dst_idx])
        
        if len(edges) == 0:
            # Single stop - no edges
            return torch.zeros((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def extract_node_features(self, trip_df, stop_sequence):
        """
        Extract features for each stop in the sequence
        
        Args:
            trip_df: GPS data for this trip
            stop_sequence: List of stop_ids
            
        Returns:
            node_features: [num_nodes, num_features] tensor
        """
        node_features = []
        
        for stop_id in stop_sequence:
            # Get GPS points near this stop
            stop_data = trip_df[trip_df['stop_id'] == stop_id]
            
            if len(stop_data) > 0:
                # Extract features from GPS data
                features = [
                    stop_data['speed'].mean(),                    # Average speed approaching stop
                    stop_data['dist_to_stop_m'].min(),           # Minimum distance to stop
                    stop_data['hour'].iloc[0],                   # Hour of day
                    stop_data['is_peak_hour'].iloc[0],           # Peak hour flag
                    stop_data['is_weekend'].iloc[0],             # Weekend flag
                    stop_data['speed_ma_1min'].mean(),           # Moving average speed
                    stop_data['latitude'].iloc[0],               # Stop location
                    stop_data['longitude'].iloc[0],
                ]
            else:
                # No data for this stop - use defaults
                features = [0.0] * 8
            
            node_features.append(features)
        
        return torch.tensor(node_features, dtype=torch.float)
    
    def trip_to_graph(self, trip_id, target_stop_id):
        """
        Convert a trip to a graph for predicting ETA to target stop
        
        Args:
            trip_id: Trip identifier
            target_stop_id: Stop we're predicting ETA to
            
        Returns:
            Data: PyTorch Geometric graph object
        """
        # Get trip data
        trip_df = self.df[self.df['trip_id'] == trip_id].copy()
        
        if len(trip_df) == 0:
            return None
        
        # Get stop sequence
        stop_sequence = self.get_stop_sequence(trip_id)
        
        if target_stop_id not in stop_sequence:
            return None
        
        # Create LOCAL node index mapping for this graph
        # Maps stop_id -> node_index (0, 1, 2, ..., len(stop_sequence)-1)
        local_stop_to_idx = {stop_id: idx for idx, stop_id in enumerate(stop_sequence)}
        
        # Build edges using LOCAL indices
        edges = []
        for i in range(len(stop_sequence) - 1):
            src_idx = i  # Local index
            dst_idx = i + 1  # Local index
            edges.append([src_idx, dst_idx])
        
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Extract node features
        x = self.extract_node_features(trip_df, stop_sequence)
        
        # Get target (ETA to target stop)
        target_data = trip_df[trip_df['stop_id'] == target_stop_id]
        if len(target_data) == 0:
            return None
        
        y = torch.tensor([target_data['ETA_sec'].iloc[0]], dtype=torch.float)
        
        # Get target node index (LOCAL index, not global!)
        target_idx = local_stop_to_idx[target_stop_id]
        
        # Create graph
        graph = Data(
            x=x,                          # Node features [num_nodes, num_features]
            edge_index=edge_index,        # Edge connections [2, num_edges]
            y=y,                          # Target ETA [1]
            num_nodes=len(stop_sequence), # Number of nodes in this graph
            target_node=target_idx        # Which node we're predicting for (LOCAL index)
        )
        
        return graph
    
    def create_dataset(self, max_samples=None):
        """
        Create full dataset of graphs
        
        Args:
            max_samples: Maximum number of graphs to create (for testing)
            
        Returns:
            List of Data objects
        """
        graphs = []
        
        # Get all (trip_id, stop_id) pairs
        samples = self.df.groupby(['trip_id', 'stop_id']).first().reset_index()[['trip_id', 'stop_id']]
        
        if max_samples:
            samples = samples.head(max_samples)
        
        print(f"\n🔨 Creating {len(samples)} graphs...")
        
        for idx, row in tqdm(samples.iterrows(), total=len(samples)):
            trip_id = row['trip_id']
            stop_id = row['stop_id']
            
            graph = self.trip_to_graph(trip_id, stop_id)
            
            if graph is not None:
                graphs.append(graph)
        
        print(f"✓ Created {len(graphs)} valid graphs")
        
        return graphs


# Test the graph builder
if __name__ == "__main__":
    print("="*80)
    print("TESTING GRAPH BUILDER")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('training_full_18days_parallel.csv')
    
    # Create builder
    builder = BusGraphBuilder(df)
    
    # Test on one trip
    print("\n🧪 Testing on sample trip...")
    trip_id = df['trip_id'].iloc[0]
    stop_id = df['stop_id'].iloc[100]
    
    graph = builder.trip_to_graph(trip_id, stop_id)
    
    if graph:
        print(f"\n✓ Sample Graph:")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Node features shape: {graph.x.shape}")
        print(f"  Edges: {graph.edge_index.shape[1]}")
        print(f"  Target ETA: {graph.y.item():.1f} seconds ({graph.y.item()/60:.1f} minutes)")
        print(f"  Target node: {graph.target_node}")
    
    # Create small dataset for testing
    print("\n🔨 Creating test dataset (100 samples)...")
    test_graphs = builder.create_dataset(max_samples=100)
    
    print(f"\n✓ Test dataset created: {len(test_graphs)} graphs")
    print(f"  Average nodes per graph: {np.mean([g.num_nodes for g in test_graphs]):.1f}")
    print(f"  Average edges per graph: {np.mean([g.edge_index.shape[1] for g in test_graphs]):.1f}")
    
    print("\n" + "="*80)
    print("✅ GRAPH BUILDER WORKING!")
    print("="*80)
