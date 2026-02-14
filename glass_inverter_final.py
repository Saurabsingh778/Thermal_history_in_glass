import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. DATA PROCESSING (Sanitized) ---
print("Loading 'glass_challenge_data.pkl'...")
with open('glass_challenge_data.pkl', 'rb') as f:
    raw_data, raw_labels = pickle.load(f)

print("Sanitizing and converting data...")
data_list = []
cutoff = 2.5 

# Collect all features first to calculate statistics for normalization
all_features = []
valid_indices = []

for i, item in enumerate(raw_data):
    feats = item['features']
    # Check for NaNs or Infs
    if np.isnan(feats).any() or np.isinf(feats).any():
        # Replace Inf with finite large number, NaN with 0
        feats = np.nan_to_num(feats, posinf=100.0, neginf=-100.0)
    
    all_features.append(feats)
    valid_indices.append(i)

# Normalize features (Crucial for GNN stability)
scaler = StandardScaler()
# Stack all particles from all graphs to fit scaler
all_features_stacked = np.vstack(all_features)
scaler.fit(all_features_stacked)
print(f"Feature Mean: {scaler.mean_}, Scale: {scaler.scale_}")

# Build Graphs
for i in valid_indices:
    item = raw_data[i]
    # Clean features again just to be safe
    raw_feats = np.nan_to_num(item['features'], posinf=100.0, neginf=-100.0)
    # Normalize
    norm_feats = scaler.transform(raw_feats)
    
    pos = torch.tensor(item['positions'], dtype=torch.float)
    x = torch.tensor(norm_feats, dtype=torch.float)
    y = torch.tensor([raw_labels[i]], dtype=torch.float)

    # Build Edges
    dist_matrix = torch.cdist(pos, pos)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    row, col = edge_index
    edge_attr = dist_matrix[row, col].unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
    data_list.append(data)

# Split
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True) # Batch size 8 for stability
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

print(f"Graph Construction Complete. Train: {len(train_data)}, Test: {len(test_data)}")

# --- 2. THE MODEL (Robust Version) ---
class GlassInverter(torch.nn.Module):
    def __init__(self):
        super(GlassInverter, self).__init__()
        
        self.encoder = nn.Linear(3, 64)
        
        # GATv2 layers
        self.gat1 = GATv2Conv(64, 64, heads=4, edge_dim=1, concat=True)
        self.gat2 = GATv2Conv(256, 64, heads=4, edge_dim=1, concat=True)
        
        self.post_mp = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128), # LayerNorm helps training dynamics
            nn.ReLU(),
            nn.Dropout(0.2),   # Dropout prevents overfitting
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.encoder(x))
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        x = global_mean_pool(x, data.batch)
        x = self.post_mp(x)
        
        # OUTPUT LOGITS (No Sigmoid here!)
        # This works better with BCEWithLogitsLoss
        out = self.classifier(x)
        return out, x

# --- 3. TRAINING LOOP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GlassInverter().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4) # AdamW is better
criterion = nn.BCEWithLogitsLoss() # Stable Loss

print(f"\nStarting Robust Training on {device}...")

best_acc = 0
for epoch in range(100): 
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out, _ = model(batch)
        loss = criterion(out.view(-1), batch.y)
        
        loss.backward()
        # Gradient Clipping (Prevents explosions)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 5 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_logits, _ = model(batch)
                # Convert logits to probabilities for accuracy check
                probs = torch.sigmoid(pred_logits)
                predicted_label = (probs > 0.5).float()
                correct += (predicted_label.view(-1) == batch.y).sum().item()
                total += batch.y.size(0)
        
        acc = 100 * correct / total
        print(f"Epoch {epoch}: Loss = {total_loss:.4f} | Test Accuracy = {acc:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'glass_inverter_best.pth')

print(f"\nFinal Best Test Accuracy: {best_acc}%")
if best_acc > 85:
    print("SUCCESS: The Geometric Inversion of Entropy is confirmed.")