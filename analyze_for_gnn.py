"""
Day 1: Analyze data and design graph structure
Quick analysis to understand what we're working with
"""
import pandas as pd
import numpy as np

print("="*80)
print("DATA ANALYSIS FOR GNN")
print("="*80)

# Load data
df = pd.read_csv('training_full_18days_parallel.csv')

print(f"\n📊 Dataset Overview:")
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {df.shape[1]}")

print(f"\n🚌 Bus System Structure:")
print(f"  Unique trips: {df['trip_id'].nunique():,}")
print(f"  Unique routes: {df['route_id'].nunique()}")
print(f"  Unique stops: {df['stop_id'].nunique()}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

print(f"\n📍 Stop Analysis:")
stops_per_route = df.groupby('route_id')['stop_id'].nunique()
print(f"  Stops per route: {stops_per_route.describe()[['mean', 'min', 'max']]}")

print(f"\n🔗 Graph Structure (Stop-Based):")
print(f"  Nodes (stops): {df['stop_id'].nunique()}")

# Calculate edges (stop connections)
edges = set()
for trip_id in df['trip_id'].unique()[:100]:  # Sample 100 trips
    trip_stops = df[df['trip_id'] == trip_id].sort_values('stop_sequence')['stop_id'].tolist()
    for i in range(len(trip_stops) - 1):
        edges.add((trip_stops[i], trip_stops[i+1]))

print(f"  Edges (connections): ~{len(edges) * (df['trip_id'].nunique() / 100):.0f} (estimated)")

print(f"\n📈 Training Examples:")
# Each (trip_id, stop_id) pair is one training example
examples = df.groupby(['trip_id', 'stop_id']).size()
print(f"  Total examples: {len(examples):,}")
print(f"  Examples per trip: {len(examples) / df['trip_id'].nunique():.1f}")

print(f"\n✅ Graph Design Decision:")
print(f"  Type: Stop-based graph (simpler than segment-based)")
print(f"  Nodes: {df['stop_id'].nunique()} stops")
print(f"  Node features: speed, distance, time, historical patterns")
print(f"  Edges: Sequential stop connections")
print(f"  Graphs to create: ~{len(examples):,}")

print("\n" + "="*80)
