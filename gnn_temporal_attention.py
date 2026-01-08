"""
Temporal Graph Attention Network (TGAT)
Advanced GNN with multi-head attention for maximum accuracy
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class TemporalGraphAttention(nn.Module):
    """
    Advanced GNN combining:
    - Multi-head attention (8 heads)
    - Temporal encoding (time-aware)
    - Skip connections (ResNet-style)
    - Dual pooling (mean + max)
    - Deep prediction head
    
    Target: Maximum accuracy for ETA prediction
    """
    def __init__(self, num_features=8, hidden=128, heads=8, dropout=0.3):
        super().__init__()
        
        # Temporal encoding layer
        # Converts hour (0-23) into rich temporal representation
        self.temporal_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Multi-head Graph Attention layers
        # Layer 1: Initial feature extraction
        self.gat1 = GATConv(
            num_features + 32,  # Input: original features + temporal
            hidden, 
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            concat=True
        )
        
        # Layer 2: Deep feature learning
        self.gat2 = GATConv(
            hidden * heads,
            hidden,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
            concat=True
        )
        
        # Layer 3: Final refinement
        self.gat3 = GATConv(
            hidden * heads,
            hidden,
            heads=4,  # Fewer heads for final layer
            dropout=dropout,
            add_self_loops=True,
            concat=True
        )
        
        # Skip connections (ResNet-style)
        # skip1: (num_features + 32) -> (hidden * heads) = 40 -> 1024
        # skip2: (hidden * heads) -> (hidden * heads) = 1024 -> 1024
        self.skip1 = nn.Linear(num_features + 32, hidden * heads)
        self.skip2 = nn.Linear(hidden * heads, hidden * heads)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden * heads)  # 1024
        self.bn2 = nn.BatchNorm1d(hidden * heads)  # 1024
        self.bn3 = nn.BatchNorm1d(hidden * 4)      # 512
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Deep prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden * 4 * 2, 256),  # *2 for dual pooling
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, 8]
                - edge_index: Graph edges [2, num_edges]
                - batch: Batch assignment [num_nodes]
        
        Returns:
            Predicted ETA (seconds)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Extract hour for temporal encoding
        hour = x[:, 2:3]  # Hour is 3rd feature
        
        # Temporal encoding
        temporal = self.temporal_embed(hour)
        
        # Concatenate original features with temporal encoding
        x = torch.cat([x, temporal], dim=1)
        
        # Store for skip connection
        x_input = x
        
        # Layer 1 with skip connection
        x1 = self.gat1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout(x1)
        
        # Skip connection from input
        x1 = x1 + self.skip1(x_input)
        
        # Layer 2 with skip connection
        x2 = self.gat2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = torch.relu(x2)
        x2 = self.dropout(x2)
        
        # Skip connection from layer 1
        x2 = x2 + self.skip2(x1)
        
        # Layer 3 (final)
        x3 = self.gat3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = torch.relu(x3)
        
        # Dual pooling: combine mean and max
        mean_pool = global_mean_pool(x3, batch)
        max_pool = global_max_pool(x3, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        
        # Final prediction
        eta = self.predictor(pooled)
        
        return eta.squeeze()
    
    def get_features(self, data):
        """
        Extract learned features (for ensemble)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        hour = x[:, 2:3]
        temporal = self.temporal_embed(hour)
        x = torch.cat([x, temporal], dim=1)
        
        x1 = self.gat1(x, edge_index)
        x1 = torch.relu(x1) + self.skip1(x)
        
        x2 = self.gat2(x1, edge_index)
        x2 = torch.relu(x2) + self.skip2(x1)
        
        x3 = self.gat3(x2, edge_index)
        x3 = torch.relu(x3)
        
        mean_pool = global_mean_pool(x3, batch)
        max_pool = global_max_pool(x3, batch)
        features = torch.cat([mean_pool, max_pool], dim=1)
        
        return features


# Test the model
if __name__ == "__main__":
    print("="*80)
    print("TESTING TEMPORAL GRAPH ATTENTION NETWORK")
    print("="*80)
    
    # Load one graph for testing
    graphs = torch.load('bus_graphs_full.pt', weights_only=False)
    test_graph = graphs[0]
    
    # Create model
    model = TemporalGraphAttention(num_features=8, hidden=128, heads=8)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Layers: 3 GAT layers + prediction head")
    print(f"  Attention heads: 8 → 8 → 4")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        test_graph.batch = torch.zeros(test_graph.num_nodes, dtype=torch.long)
        
        pred = model(test_graph)
        actual = test_graph.y.item()
        
        print(f"\n🧪 Test Prediction:")
        print(f"  Predicted ETA: {pred.item():.1f} seconds ({pred.item()/60:.1f} minutes)")
        print(f"  Actual ETA: {actual:.1f} seconds ({actual/60:.1f} minutes)")
        print(f"  Error: {abs(pred.item() - actual):.1f} seconds ({abs(pred.item() - actual)/60:.1f} minutes)")
    
    print("\n" + "="*80)
    print("✅ TGAT MODEL READY!")
    print("="*80)
    print("\nNext: Train the model on full dataset")
