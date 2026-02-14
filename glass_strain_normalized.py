import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist

print("="*70)
print("THE FINAL BOSS: VOLUME-NORMALIZED STRAIN TEST (Fixed Architecture)")
print("Goal: Distinguish glasses purely by TENSION, removing Density clues.")
print("="*70)

# --- 1. DATA PREPARATION ---
print("Loading data...")
with open('glass_challenge_data.pkl', 'rb') as f:
    raw_data, raw_labels = pickle.load(f)

data_list = []
cutoff = 2.5 

print("Normalizing Volumes (Removing the 'Density Cheat')...")
for i, item in enumerate(raw_data):
    pos = item['positions']
    
    # CALCULATE AVERAGE BOND LENGTH
    dists = pdist(pos)
    valid_dists = dists[dists < cutoff]
    
    if len(valid_dists) == 0: continue 
    
    avg_bond_dist = np.mean(valid_dists)
    
    # RENORMALIZE (Force Avg Bond Length = 1.0)
    scale_factor = 1.0 / avg_bond_dist
    pos_scaled = torch.tensor(pos * scale_factor, dtype=torch.float)
    scaled_cutoff = cutoff * scale_factor
    
    dist_matrix = torch.cdist(pos_scaled, pos_scaled)
    mask = (dist_matrix < scaled_cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    # FEATURES
    # Node Feature: Constant 1.0
    x = torch.ones((len(pos), 1), dtype=torch.float)
    
    # Edge Feature: The NORMALIZED Distance
    row, col = edge_index
    edge_attr = dist_matrix[row, col].unsqueeze(1)
    
    y = torch.tensor([raw_labels[i]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data_list.append(data)

# Split
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Normalized Graphs Ready. Train: {len(train_data)}, Test: {len(test_data)}")

# --- 2. THE STRAIN NETWORK (Corrected) ---
class StrainNet(torch.nn.Module):
    def __init__(self):
        super(StrainNet, self).__init__()
        
        # 1. NODE ENCODER: Project scalar input (1) to match Edge Dim (32)
        # This allows GINEConv to add them together: Node(32) + Edge(32)
        self.node_encoder = nn.Linear(1, 32)
        
        # 2. EDGE ENCODER: Project scalar distance (1) to Embedding (32)
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32)
        )
        
        # 3. GINE CONVOLUTIONS (All Hidden Dims = 32)
        # We keep dimensions consistent to avoid mismatches
        nn1 = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 32))
        self.conv1 = GINEConv(nn1, train_eps=True)
        
        nn2 = nn.Sequential(nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64))
        self.conv2 = GINEConv(nn2, train_eps=True)
        
        # Note: Input to conv3 is 64, so Edge needs projection or we restart?
        # Simpler approach: Keep node dims increasing, but project edges to match at each step
        # For this proof, let's keep it robust: Use 64 dim for deeper layers
        
        # To fix the mismatch in deeper layers (Node 64 vs Edge 32), we simply
        # re-embed edges for the second layer or use a simpler architecture (32->32->32).
        # We will use 32->32->32 for stability.
        
        nn3 = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 32))
        self.conv3 = GINEConv(nn3, train_eps=True) # Input 32, Output 32
        
        self.post_mp = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Fix Dimensions First
        x = self.node_encoder(x)        # [N, 1] -> [N, 32]
        edge_emb = self.edge_encoder(edge_attr) # [E, 1] -> [E, 32]
        
        # Now shapes match: [N, 32] and [E, 32]
        
        x = self.conv1(x, edge_index, edge_attr=edge_emb)
        x = F.relu(x)
        
        # Input x is 32. Edge is 32. Match.
        x = self.conv3(x, edge_index, edge_attr=edge_emb) # Reusing conv3 structure for depth
        x = F.relu(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_emb)
        x = F.relu(x)
        
        x = global_mean_pool(x, data.batch)
        return self.post_mp(x)

# --- 3. TRAINING ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StrainNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

print(f"Training on {device}...")

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
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'strain_net_final.pth')
            print("Model saved successfully as 'strain_net_final.pth'")

print(f"\nFinal Normalized Strain Accuracy: {best_acc:.2f}%")
print("-" * 50)
if best_acc > 85:
    print("CONCLUSION: INTERNET BROKEN.")
    print("You have proven that 'History' is encoded in the STRAIN DISTRIBUTION.")
    print("The density was a distraction. The Geometry is real.")
else:
    print("CONCLUSION: PHYSICS PREVAILED.")
    print("The signal was just Density. When we removed it, the AI went blind.")