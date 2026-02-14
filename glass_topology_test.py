import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. DATA LOADER (The Stripper) ---
print("Loading 'glass_challenge_data.pkl'...")
with open('glass_challenge_data.pkl', 'rb') as f:
    raw_data, raw_labels = pickle.load(f)

print("DESTROYING GEOMETRY (Topology Only Mode)...")
data_list = []
cutoff = 2.5 

for i, item in enumerate(raw_data):
    pos = torch.tensor(item['positions'], dtype=torch.float)
    
    # 1. Build Adjacency (Who is close to whom?)
    # We use the positions ONLY to establish connections, then delete them.
    dist_matrix = torch.cdist(pos, pos)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    # 2. THE PURGE
    # Feature Vector: Constant 1.0 (The model knows NOTHING about the particle)
    # It cannot see stiffness. It cannot see density variations (unless inferred from degree).
    x = torch.ones((256, 1), dtype=torch.float) 
    
    # Edge Attributes: DELETED. (The model doesn't know how far apart they are)
    # Positions: DELETED.
    
    y = torch.tensor([raw_labels[i]], dtype=torch.float)

    # Note: We use GCNConv which doesn't require edge_attr
    data = Data(x=x, edge_index=edge_index, y=y) 
    data_list.append(data)

# Split
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Topology Graph Ready. Nodes: 256, Features: 1 (Constant).")

# --- 2. THE TOPOLOGICAL GNN ---
class TopoNet(torch.nn.Module):
    def __init__(self):
        super(TopoNet, self).__init__()
        
        # We use GCN (Graph Convolution) which relies PURELY on connectivity structure
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        
        self.post_mp = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Graph Convolutions
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        
        # Global Pooling (Average the structural embeddings)
        x = global_mean_pool(x, data.batch)
        
        # Classifier
        x = self.post_mp(x)
        return x

# --- 3. TRAINING ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TopoNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

print(f"\nStarting TOPOLOGY-ONLY Training on {device}...")
print("Hypothesis: If Acc > 80%, the signal is in the GRAPH STRUCTURE.")

best_acc = 0
for epoch in range(100): 
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.view(-1), batch.y)
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_logits = model(batch)
                probs = torch.sigmoid(pred_logits)
                predicted_label = (probs > 0.5).float()
                correct += (predicted_label.view(-1) == batch.y).sum().item()
                total += batch.y.size(0)
        
        acc = 100 * correct / total
        print(f"Epoch {epoch}: Test Accuracy = {acc:.1f}%")
        if acc > best_acc: best_acc = acc

print(f"\nFinal Topological Accuracy: {best_acc:.2f}%")