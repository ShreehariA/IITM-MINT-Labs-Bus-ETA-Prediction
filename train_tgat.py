"""
Training script for Temporal Graph Attention Network
Run this in your notebook to train the model
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from gnn_temporal_attention import TemporalGraphAttention
import numpy as np
from tqdm import tqdm

print("="*80)
print("TRAINING TEMPORAL GRAPH ATTENTION NETWORK")
print("="*80)

# 1. Load graph dataset
print("\n📂 Loading graph dataset...")
graphs = torch.load('bus_graphs_full.pt', weights_only=False)
print(f"  Total graphs: {len(graphs)}")

# 2. Split into train/val/test
train_size = int(0.7 * len(graphs))
val_size = int(0.15 * len(graphs))

train_graphs = graphs[:train_size]
val_graphs = graphs[train_size:train_size+val_size]
test_graphs = graphs[train_size+val_size:]

print(f"\n📊 Dataset split:")
print(f"  Train: {len(train_graphs)} graphs ({len(train_graphs)/len(graphs)*100:.1f}%)")
print(f"  Val:   {len(val_graphs)} graphs ({len(val_graphs)/len(graphs)*100:.1f}%)")
print(f"  Test:  {len(test_graphs)} graphs ({len(test_graphs)/len(graphs)*100:.1f}%)")

# 3. Create data loaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32)
test_loader = DataLoader(test_graphs, batch_size=32)

print(f"\n🔄 Batches per epoch: {len(train_loader)}")

# 4. Initialize model
print("\n🧠 Initializing model...")
model = TemporalGraphAttention(num_features=8, hidden=128, heads=8, dropout=0.3)

num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,}")

# 5. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=10)
criterion = nn.MSELoss()

# 6. Training loop
print("\n🚀 Starting training...")
print("="*80)

best_val_loss = float('inf')
patience_counter = 0
max_patience = 20

for epoch in range(100):
    # Training
    model.train()
    train_loss = 0
    train_preds = []
    train_actuals = []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/100"):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.extend(pred.detach().cpu().numpy())
        train_actuals.extend(batch.y.cpu().numpy())
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_actuals = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            
            val_loss += loss.item()
            val_preds.extend(pred.cpu().numpy())
            val_actuals.extend(batch.y.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Calculate accuracy metrics
    train_preds = np.array(train_preds)
    train_actuals = np.array(train_actuals)
    val_preds = np.array(val_preds)
    val_actuals = np.array(val_actuals)
    
    train_acc_1min = (np.abs(train_preds - train_actuals) <= 60).mean() * 100
    val_acc_1min = (np.abs(val_preds - val_actuals) <= 60).mean() * 100
    val_acc_5min = (np.abs(val_preds - val_actuals) <= 300).mean() * 100
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Print progress
    print(f"\nEpoch {epoch+1}:")
    print(f"  Train Loss: {train_loss:.2f} | Train ±1min: {train_acc_1min:.1f}%")
    print(f"  Val Loss:   {val_loss:.2f} | Val ±1min: {val_acc_1min:.1f}% | Val ±5min: {val_acc_5min:.1f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'tgat_best.pt')
        print(f"  ✓ New best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"\n⚠️  Early stopping after {epoch+1} epochs")
            break

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)

# 7. Final evaluation on test set
print("\n📊 Final Evaluation on Test Set...")
model.load_state_dict(torch.load('tgat_best.pt'))
model.eval()

test_preds = []
test_actuals = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        test_preds.extend(pred.cpu().numpy())
        test_actuals.extend(batch.y.cpu().numpy())

test_preds = np.array(test_preds)
test_actuals = np.array(test_actuals)

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

print("\n💾 Model saved to: tgat_best.pt")
print("="*80)
