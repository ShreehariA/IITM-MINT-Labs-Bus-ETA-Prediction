"""
Create full graph dataset for GNN training
This will take ~5-10 minutes
"""
import torch
import pandas as pd
from graph_builder import BusGraphBuilder

print("="*80)
print("CREATING FULL GRAPH DATASET")
print("="*80)

# Load data
print("\n📂 Loading GPS data...")
df = pd.read_csv('training_full_18days_parallel.csv')
print(f"  Loaded {len(df):,} GPS points")

# Create builder
print("\n🔧 Initializing graph builder...")
builder = BusGraphBuilder(df)

# Create full dataset
print("\n🔨 Creating all graphs (this will take ~5-10 minutes)...")
graphs = builder.create_dataset()

# Save dataset
print("\n💾 Saving dataset...")
torch.save(graphs, 'bus_graphs_full.pt')

print(f"\n✅ Dataset saved: bus_graphs_full.pt")
print(f"  Total graphs: {len(graphs)}")
print(f"  File size: {os.path.getsize('bus_graphs_full.pt') / 1024 / 1024:.1f} MB")

# Print statistics
import numpy as np
print(f"\n📊 Dataset Statistics:")
print(f"  Graphs: {len(graphs)}")
print(f"  Avg nodes per graph: {np.mean([g.num_nodes for g in graphs]):.1f}")
print(f"  Avg edges per graph: {np.mean([g.edge_index.shape[1] for g in graphs]):.1f}")
print(f"  Avg ETA: {np.mean([g.y.item() for g in graphs])/60:.1f} minutes")

print("\n" + "="*80)
print("✅ DAY 1 COMPLETE!")
print("="*80)
print("\nNext: Day 2 - Implement and train GNN model")
