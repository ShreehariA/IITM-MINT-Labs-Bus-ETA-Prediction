"""
Fixed GNN Training with Proper Normalization
This will solve the catastrophic failure
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from gnn_temporal_attention import TemporalGraphAttention
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle

print("="*80)
print("TRAINING TGAT WITH PROPER NORMALIZATION")
print("="*80)

# 1. Load graph dataset
print("\n📂 Loading graph dataset...")
graphs = torch.load('bus_graphs_full.pt', weights_only=False)
print(f"  Total graphs: {len(graphs)}")

# 2. Calculate normalization statistics from TRAINING data only
print("\n📊 Calculating normalization statistics...")

train_size = int(0.7 * len(graphs))
train_graphs = graphs[:train_size]

# Collect all features and targets from training set
all_features = []
all_targets = []

for g in train_graphs:
    all_features.append(g.x.numpy())
    all_targets.append(g.y.item())

all_features = np.vstack(all_features)
all_targets = np.array(all_targets)

# Feature normalization (StandardScaler)
feature_scaler = StandardScaler()
feature_scaler.fit(all_features)

# Target normalization (simple min-max to 0-1)
target_min = all_targets.min()
target_max = all_targets.max()
target_range = target_max - target_min

print(f"\n  Feature stats:")
print(f"    Mean: {feature_scaler.mean_[:3]}... (first 3)")
print(f"    Std: {feature_scaler.scale_[:3]}... (first 3)")
print(f"\n  Target stats:")
print(f"    Min: {target_min:.1f} seconds")
print(f"    Max: {target_max:.1f} seconds")
print(f"    Range: {target_range:.1f} seconds")

# Save scalers for later use
with open('scalers.pkl', 'wb') as f:
    pickle.dump({
        'feature_scaler': feature_scaler,
        'target_min': target_min,
        'target_max': target_max,
        'target_range': target_range
    }, f)

# 3. Normalize ALL graphs
print("\n🔄 Normalizing all graphs...")
for g in tqdm(graphs):
    # Normalize features
    g.x = torch.tensor(feature_scaler.transform(g.x.numpy()), dtype=torch.float)
    
    # Normalize target to [0, 1]
    g.y = (g.y - target_min) / target_range

# 4. Split into train/val/test
val_size = int(0.15 * len(graphs))

train_graphs = graphs[:train_size]
val_graphs = graphs[train_size:train_size+val_size]
test_graphs = graphs[train_size+val_size:]

print(f"\n📊 Dataset split:")
print(f"  Train: {len(train_graphs)} graphs")
print(f"  Val:   {len(val_graphs)} graphs")
print(f"  Test:  {len(test_graphs)} graphs")

# 5. Create data loaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)
test_loader = DataLoader(test_graphs, batch_size=32)

# 6. Initialize model
print("\n🧠 Initializing model...")
model = TemporalGraphAttention(num_features=8, hidden=128, heads=8, dropout=0.3)
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=10)
criterion = nn.MSELoss()

# 7. Training loop
print("\n🚀 Starting training...")
print("="*80)

best_val_loss = float('inf')
patience_counter = 0
max_patience = 20

for epoch in range(100):
    # Training
    model.train()
    train_loss = 0
    train_preds_norm = []
    train_actuals_norm = []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/100"):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        
        # Gradient clipping (important!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        train_preds_norm.extend(pred.detach().cpu().numpy())
        train_actuals_norm.extend(batch.y.cpu().numpy())
    
    train_loss /= len(train_loader)
    
    # Denormalize for metrics
    train_preds = np.array(train_preds_norm) * target_range + target_min
    train_actuals = np.array(train_actuals_norm) * target_range + target_min
    
    # Validation
    model.eval()
    val_loss = 0
    val_preds_norm = []
    val_actuals_norm = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            
            val_loss += loss.item()
            val_preds_norm.extend(pred.cpu().numpy())
            val_actuals_norm.extend(batch.y.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Denormalize for metrics
    val_preds = np.array(val_preds_norm) * target_range + target_min
    val_actuals = np.array(val_actuals_norm) * target_range + target_min
    
    # Calculate accuracy metrics (in original scale)
    train_acc_1min = (np.abs(train_preds - train_actuals) <= 60).mean() * 100
    val_acc_1min = (np.abs(val_preds - val_actuals) <= 60).mean() * 100
    val_acc_5min = (np.abs(val_preds - val_actuals) <= 300).mean() * 100
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Print progress
    print(f"\nEpoch {epoch+1}:")
    print(f"  Train Loss: {train_loss:.4f} | Train ±1min: {train_acc_1min:.1f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val ±1min: {val_acc_1min:.1f}% | Val ±5min: {val_acc_5min:.1f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'tgat_best_normalized.pt')
        print(f"  ✓ New best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\n⚠️  Early stopping after {epoch+1} epochs")
            break

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)

# 8. Final evaluation on test set
print("\n📊 Final Evaluation on Test Set...")
model.load_state_dict(torch.load('tgat_best_normalized.pt', weights_only=False))
model.eval()

test_preds_norm = []
test_actuals_norm = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        test_preds_norm.extend(pred.cpu().numpy())
        test_actuals_norm.extend(batch.y.cpu().numpy())

# Denormalize
test_preds = np.array(test_preds_norm) * target_range + target_min
test_actuals = np.array(test_actuals_norm) * target_range + target_min

# Calculate metrics
mae = np.mean(np.abs(test_preds - test_actuals)) / 60  # minutes
rmse = np.sqrt(np.mean((test_preds - test_actuals)**2)) / 60
acc_1min = (np.abs(test_preds - test_actuals) <= 60).mean() * 100
acc_2min = (np.abs(test_preds - test_actuals) <= 120).mean() * 100
acc_5min = (np.abs(test_preds - test_actuals) <= 300).mean() * 100

print(f"\n📈 Test Set Results:")
print(f"  MAE:  {mae:.2f} minutes")
print(f"  RMSE: {rmse:.2f} minutes")
print(f"  ±1 min accuracy: {acc_1min:.1f}%")
print(f"  ±2 min accuracy: {acc_2min:.1f}%")
print(f"  ±5 min accuracy: {acc_5min:.1f}%")

print("\n💾 Model saved to: tgat_best_normalized.pt")
print("💾 Scalers saved to: scalers.pkl")
print("="*80)
