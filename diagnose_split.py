"""
Diagnose why validation is 87% but test is 23%
This suggests a serious train/test split issue
"""
import torch
import numpy as np
import pandas as pd

print("="*80)
print("DIAGNOSING TRAIN/VAL/TEST SPLIT ISSUE")
print("="*80)

# Load graphs
graphs = torch.load('bus_graphs_full.pt', weights_only=False)

# Split (same as training)
train_size = int(0.7 * len(graphs))
val_size = int(0.15 * len(graphs))

train_graphs = graphs[:train_size]
val_graphs = graphs[train_size:train_size+val_size]
test_graphs = graphs[train_size+val_size:]

print(f"\n📊 Split sizes:")
print(f"  Train: {len(train_graphs)}")
print(f"  Val:   {len(val_graphs)}")
print(f"  Test:  {len(test_graphs)}")

# Check target distributions
train_targets = [g.y.item() for g in train_graphs]
val_targets = [g.y.item() for g in val_graphs]
test_targets = [g.y.item() for g in test_graphs]

print(f"\n🎯 Target Distribution (BEFORE normalization):")
print(f"  Train: mean={np.mean(train_targets):.1f}s, std={np.std(train_targets):.1f}s")
print(f"  Val:   mean={np.mean(val_targets):.1f}s, std={np.std(val_targets):.1f}s")
print(f"  Test:  mean={np.mean(test_targets):.1f}s, std={np.std(test_targets):.1f}s")

# Load original data to check trip_ids
df = pd.read_csv('training_full_18days_parallel.csv')

print(f"\n📅 Original Data:")
print(f"  Total trips: {df['trip_id'].nunique()}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

# Check if graphs are ordered by date or random
print(f"\n🔍 Checking graph ordering...")

# Get first and last graphs' trip IDs
# (Need to trace back to original data)
print(f"  First graph target: {graphs[0].y.item():.1f}s")
print(f"  Middle graph target: {graphs[len(graphs)//2].y.item():.1f}s")
print(f"  Last graph target: {graphs[-1].y.item():.1f}s")

# Check if there's a pattern
train_mean = np.mean(train_targets)
val_mean = np.mean(val_targets)
test_mean = np.mean(test_targets)

print(f"\n⚠️  PROBLEM DETECTED:")
if abs(test_mean - train_mean) > 200:
    print(f"  Test set has VERY different distribution!")
    print(f"  Train mean: {train_mean:.1f}s")
    print(f"  Test mean:  {test_mean:.1f}s")
    print(f"  Difference: {abs(test_mean - train_mean):.1f}s")
    print(f"\n  This suggests:")
    print(f"  - Graphs are ordered by time/date")
    print(f"  - Test set is from different time period")
    print(f"  - Need RANDOM split, not sequential!")

# Check how graphs were created
print(f"\n🔨 How were graphs created?")
print(f"  Graph builder used groupby(['trip_id', 'stop_id'])")
print(f"  This preserves original data order")
print(f"  If original data is sorted by date, graphs are too!")

print("\n" + "="*80)
print("SOLUTION: Shuffle graphs before splitting!")
print("="*80)
