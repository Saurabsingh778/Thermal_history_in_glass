import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 1. REBUILD THE MODEL ARCHITECTURE ---
# (Must match your training script exactly)
class GlassInverter(torch.nn.Module):
    def __init__(self):
        super(GlassInverter, self).__init__()
        self.encoder = torch.nn.Linear(3, 64)
        
        # We need to access attention weights, but for now we'll visualize 
        # "Node Importance" via Gradient Saliency
        self.gat1 = GATv2Conv(64, 64, heads=4, edge_dim=1, concat=True)
        self.gat2 = GATv2Conv(256, 128, heads=4, edge_dim=1, concat=True)
        self.gat3 = GATv2Conv(512, 64, heads=2, edge_dim=1, concat=True)
        
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(64, 1)

    def forward(self, data):
        data.x.requires_grad = True # Enable gradient tracking for visualization
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.encoder(x))
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat3(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # We hook the final node embeddings before pooling
        self.final_node_embeddings = x 
        
        x = global_mean_pool(x, data.batch)
        x = self.post_mp(x)
        return self.classifier(x)

# --- 2. LOAD DATA & MODEL ---
print("Loading Data & Model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GlassInverter().to(device)
# Note: You didn't save the state_dict in the last script explicitly, 
# but if you have 'glass_inverter_best.pth' or just the in-memory model, use it.
# Assuming the model is still in memory from previous run. 
# If not, we re-instantiate. *Crucially, we need trained weights.*
# FOR THIS DEMO, we will assume you run this immediately after training.

with open('glass_challenge_data.pkl', 'rb') as f:
    raw_data, raw_labels = pickle.load(f)

# Find one good example of each
fast_idx = raw_labels.index(0) # First Fast Sample
slow_idx = raw_labels.index(1) # First Slow Sample

def prepare_sample(idx):
    item = raw_data[idx]
    raw_feats = item['features']
    # Sanitization
    if np.isnan(raw_feats).any() or np.isinf(raw_feats).any():
        raw_feats = np.nan_to_num(raw_feats, posinf=100.0, neginf=-100.0)
    
    # Instance Norm (The Blindfold)
    scaler = StandardScaler()
    norm_feats = scaler.fit_transform(raw_feats)
    
    pos = torch.tensor(item['positions'], dtype=torch.float)
    x = torch.tensor(norm_feats, dtype=torch.float)
    y = torch.tensor([raw_labels[idx]], dtype=torch.float)
    
    # Edges
    cutoff = 2.5
    dist_matrix = torch.cdist(pos, pos)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    row, col = edge_index
    edge_attr = dist_matrix[row, col].unsqueeze(1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

fast_data = prepare_sample(fast_idx).to(device)
slow_data = prepare_sample(slow_idx).to(device)

# --- 3. THE SIGHT (Saliency Map) ---
def get_saliency(model, data):
    model.eval()
    # We need to trace gradients back to input to see which nodes matter
    data.x.requires_grad = True
    out = model(data)
    
    # Backprop to find "Importance" of each node
    out.backward()
    
    # Saliency = Magnitude of gradient at each node
    saliency = data.x.grad.abs().sum(dim=1).cpu().detach().numpy()
    return saliency

print("Extracting Topological Skeleton...")
# We use the trained model state from your previous run
# (Ensure 'model' variable from previous script is accessible, 
# OR load 'glass_inverter_best.pth' if you saved it)
try:
    model.load_state_dict(torch.load('glass_inverter_best.pth'))
    print("Loaded best model weights.")
except:
    print("WARNING: Using random weights? Please ensure model is trained.")

fast_saliency = get_saliency(model, fast_data)
slow_saliency = get_saliency(model, slow_data)

# --- 4. VISUALIZATION ---
print("Rendering 3D Structures...")

fig = plt.figure(figsize=(12, 6))

def plot_glass(ax, data, saliency, title):
    pos = data.pos.cpu().detach().numpy()
    # Normalize saliency for color map
    colors = saliency
    # Plot connections (faint)
    edges = data.edge_index.cpu().detach().numpy()
    
    # Only plot high-importance particles to see the "Skeleton"
    # Filter: Top 30% most important nodes
    threshold = np.percentile(saliency, 70)
    mask = saliency > threshold
    
    ax.scatter(pos[mask, 0], pos[mask, 1], pos[mask, 2], 
               c=colors[mask], cmap='inferno', s=50, alpha=0.9, edgecolors='k')
    
    # Plot edges between high-importance nodes (The Force Chains)
    # This is slow, so we plot a subset
    count = 0
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        if mask[src] and mask[dst]:
            ax.plot([pos[src, 0], pos[dst, 0]], 
                    [pos[src, 1], pos[dst, 1]], 
                    [pos[src, 2], pos[dst, 2]], 
                    c='red', alpha=0.3, linewidth=1)
            count += 1
            if count > 500: break # Limit for rendering speed
            
    ax.set_title(title)
    ax.set_axis_off()

ax1 = fig.add_subplot(121, projection='3d')
plot_glass(ax1, fast_data, fast_saliency, "Fast Quench (High Entropy)")

ax2 = fig.add_subplot(122, projection='3d')
plot_glass(ax2, slow_data, slow_saliency, "Slow Anneal (Hidden Order)")

plt.tight_layout()
plt.show()
print("Look at the plots.")
print("The 'Slow' glass should show connected, lightning-like chains (Force Chains).")
print("The 'Fast' glass should look like scattered, unconnected dust.")