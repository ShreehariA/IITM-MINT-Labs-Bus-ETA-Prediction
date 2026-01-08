"""
Quick diagnostic to understand what went wrong
"""
import torch
import numpy as np

print("="*80)
print("DIAGNOSING GNN FAILURE")
print("="*80)

# Load graphs
graphs = torch.load('bus_graphs_full.pt', weights_only=False)

print(f"\n📊 Data Statistics:")
print(f"  Total graphs: {len(graphs)}")

# Check target distribution
targets = [g.y.item() for g in graphs]
print(f"\n🎯 Target (ETA) Distribution:")
print(f"  Min: {min(targets):.1f} seconds ({min(targets)/60:.1f} minutes)")
print(f"  Max: {max(targets):.1f} seconds ({max(targets)/60:.1f} minutes)")
print(f"  Mean: {np.mean(targets):.1f} seconds ({np.mean(targets)/60:.1f} minutes)")
print(f"  Std: {np.std(targets):.1f} seconds ({np.std(targets)/60:.1f} minutes)")

# Check feature distribution
sample_features = graphs[0].x.numpy()
print(f"\n📈 Feature Statistics (first graph):")
print(f"  Shape: {sample_features.shape}")
print(f"  Min: {sample_features.min():.2f}")
print(f"  Max: {sample_features.max():.2f}")
print(f"  Mean: {sample_features.mean():.2f}")

# Check for NaN/Inf
has_nan = any(torch.isnan(g.x).any() or torch.isnan(g.y).any() for g in graphs[:100])
has_inf = any(torch.isinf(g.x).any() or torch.isinf(g.y).any() for g in graphs[:100])

print(f"\n⚠️  Data Quality:")
print(f"  Has NaN: {has_nan}")
print(f"  Has Inf: {has_inf}")

# Check model predictions
from gnn_temporal_attention import TemporalGraphAttention

model = TemporalGraphAttention()
model.eval()

test_graph = graphs[0]
test_graph.batch = torch.zeros(test_graph.num_nodes, dtype=torch.long)

with torch.no_grad():
    pred = model(test_graph)
    print(f"\n🔮 Untrained Model Prediction:")
    print(f"  Predicted: {pred.item():.1f} seconds")
    print(f"  Actual: {test_graph.y.item():.1f} seconds")
    print(f"  Scale issue: {abs(pred.item() / test_graph.y.item()):.2f}x")

# Load trained model
model.load_state_dict(torch.load('tgat_best.pt', weights_only=False))
model.eval()

with torch.no_grad():
    pred = model(test_graph)
    print(f"\n🔮 Trained Model Prediction:")
    print(f"  Predicted: {pred.item():.1f} seconds ({pred.item()/60:.1f} minutes)")
    print(f"  Actual: {test_graph.y.item():.1f} seconds ({test_graph.y.item()/60:.1f} minutes)")
    print(f"  Error: {abs(pred.item() - test_graph.y.item()):.1f} seconds")

print("\n" + "="*80)
