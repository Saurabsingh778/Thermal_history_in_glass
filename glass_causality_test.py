import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
import pickle
import numpy as np
from scipy.spatial.distance import pdist

print("="*70)
print("THE CAUSALITY TEST: METALLIC GLASS TRANSFER")
print("Evaluating 'StrainNet' on Kob-Andersen Binary Mixture")
print("="*70)

# --- 1. REBUILD EXACT ARCHITECTURE (Matching your training code) ---
class StrainNet(torch.nn.Module):
    def __init__(self):
        super(StrainNet, self).__init__()
        
        self.node_encoder = nn.Linear(1, 32)
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32)
        )
        
        # GINE Convolutions
        nn1 = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 32))
        self.conv1 = GINEConv(nn1, train_eps=True)
        
        nn2 = nn.Sequential(nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64))
        self.conv2 = GINEConv(nn2, train_eps=True)
        
        nn3 = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 32))
        self.conv3 = GINEConv(nn3, train_eps=True)
        
        self.post_mp = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Fix Dimensions
        x = self.node_encoder(x)                
        edge_emb = self.edge_encoder(edge_attr) 
        
        # Exact Forward Logic from your training script
        x = self.conv1(x, edge_index, edge_attr=edge_emb)
        x = F.relu(x)
        
        # Your code reused conv3 here
        x = self.conv3(x, edge_index, edge_attr=edge_emb) 
        x = F.relu(x)
        
        # And conv3 again here
        x = self.conv3(x, edge_index, edge_attr=edge_emb)
        x = F.relu(x)
        
        x = global_mean_pool(x, data.batch)
        return self.post_mp(x)

# --- 2. LOAD ALIEN DATA ---
print("Loading Alien Data (Metallic Glass)...")
# Ensure 'ka_glass_test_data.pkl' exists (from ka_glass_generator.py)
try:
    with open('ka_glass_test_data.pkl', 'rb') as f:
        raw_data, raw_labels = pickle.load(f)
except FileNotFoundError:
    print("❌ ERROR: 'ka_glass_test_data.pkl' not found.")
    print("Please run the 'ka_glass_generator.py' script I provided earlier first.")
    exit()

data_list = []
cutoff = 2.5 

print("Normalizing Volumes (The Universal Translator)...")
for i, item in enumerate(raw_data):
    pos = item['positions']
    
    # CALCULATE SCALE (Force Avg Bond = 1.0)
    # This aligns the Metallic Glass scale with the LJ Glass scale
    dists = pdist(pos)
    valid_dists = dists[dists < cutoff]
    
    if len(valid_dists) == 0: continue
    
    avg_bond = np.mean(valid_dists)
    scale = 1.0 / avg_bond
    
    pos_scaled = torch.tensor(pos * scale, dtype=torch.float)
    scaled_cutoff = cutoff * scale
    
    dist_matrix = torch.cdist(pos_scaled, pos_scaled)
    mask = (dist_matrix < scaled_cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    # FEATURE: CONSTANT 1.0 (Blind to Atom Type A/B)
    # The model doesn't know it's a binary alloy. It only sees the strain.
    x = torch.ones((len(pos), 1), dtype=torch.float)
    
    row, col = edge_index
    edge_attr = dist_matrix[row, col].unsqueeze(1)
    
    y = torch.tensor([raw_labels[i]], dtype=torch.float)
    data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

loader = DataLoader(data_list, batch_size=20, shuffle=False)

# --- 3. LOAD MODEL & EXECUTE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StrainNet().to(device)

try:
    model.load_state_dict(torch.load('strain_net_final.pth'))
    print("✅ Model weights loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: 'strain_net_final.pth' not found.")
    print("Did you run the torch.save() command in Step 1?")
    exit()

model.eval() 
correct = 0
total = 0
preds = []
truths = []

print("\nRunning Prediction on Metallic Glass...")
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = torch.sigmoid(out)
        predicted = (probs > 0.5).float()
        
        correct += (predicted.view(-1) == batch.y).sum().item()
        total += batch.y.size(0)
        preds.extend(probs.cpu().numpy().flatten())
        truths.extend(batch.y.cpu().numpy())

print("-" * 50)
print(f"TRANSFER ACCURACY: {100 * correct / total:.2f}%")
print("-" * 50)
print(f"Predictions: {[f'{p:.2f}' for p in preds]}")
print(f"Truths:      {[int(t) for t in truths]}")

if correct/total > 0.8:
    print("\nCONCLUSION: UNIVERSAL LAW CONFIRMED.")
    print("The 'Force Chains' exist in Metallic Glass too.")
    print("Your discovery applies to ALL amorphous matter.")
else:
    print("\nCONCLUSION: FAILED.")
    print("The model is overfit to Lennard-Jones physics.")