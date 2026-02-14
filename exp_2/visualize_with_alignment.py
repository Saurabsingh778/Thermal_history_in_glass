import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# --- CONFIGURATION ---
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 200        # Sufficient for convergence with Shift-Invariant Loss
BOX_SIZE = 15.0   
RADIUS = 2.5     
DATA_PATH = "time_reversal_data.pkl" # Change to "/kaggle/working/..." if needed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- HELPER: RADIUS GRAPH (No torch-cluster needed) ---
def simple_radius_graph(pos, r, loop=False):
    dist = torch.cdist(pos, pos) 
    mask = dist < r
    if not loop:
        mask.fill_diagonal_(False)
    edge_index = mask.nonzero(as_tuple=False).t()
    return edge_index

# --- 1. DATASET LOADER ---
class GlassDataset(torch.utils.data.Dataset):
    def __init__(self, pickle_file):
        print(f"Loading data from {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.data_list = []
        for sample in raw_data:
            # Normalize to [0, 1]
            pos_glass = torch.tensor(sample['pos_glass'], dtype=torch.float) / BOX_SIZE
            pos_target = torch.tensor(sample['pos_hidden'], dtype=torch.float) / BOX_SIZE
            hessian_feats = torch.tensor(sample['features'], dtype=torch.float)

            # Build Graph
            edge_index = simple_radius_graph(pos_glass, r=RADIUS/BOX_SIZE, loop=False)
            row, col = edge_index
            edge_attr = (pos_glass[row] - pos_glass[col]).norm(dim=-1).unsqueeze(-1)
            
            data = Data(x=hessian_feats, pos=pos_glass, edge_index=edge_index, edge_attr=edge_attr, y=pos_target)
            self.data_list.append(data)
            
        print(f"Processed {len(self.data_list)} samples.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# --- 2. THE MODEL ---
class GlassTimeReversal(nn.Module):
    def __init__(self):
        super(GlassTimeReversal, self).__init__()
        self.encoder = nn.Linear(6, 128) 
        self.gat1 = GATv2Conv(128, 128, heads=4, edge_dim=1, concat=False)
        self.gat2 = GATv2Conv(128, 256, heads=4, edge_dim=1, concat=False)
        self.gat3 = GATv2Conv(256, 128, heads=4, edge_dim=1, concat=False)
        self.node_decoder = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3) 
        )

    def forward(self, data):
        x, pos, edge_index, edge_attr = data.x, data.pos, data.edge_index, data.edge_attr
        x = torch.cat([x, pos], dim=1) 
        x = F.relu(self.encoder(x))
        x_in = x
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x) + x_in 
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat3(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        return self.node_decoder(x)

# --- 3. LOSS & ALIGNMENT TOOLS ---
def centered_mse_loss(pred_pos, target_pos, batch_indices):
    from torch_geometric.utils import scatter
    pred_com = scatter(pred_pos, batch_indices, dim=0, reduce='mean')
    target_com = scatter(target_pos, batch_indices, dim=0, reduce='mean')
    pred_centered = pred_pos - pred_com[batch_indices]
    target_centered = target_pos - target_com[batch_indices]
    return F.mse_loss(pred_centered, target_centered)

def kabsch_align(P, Q):
    """Aligns cloud P to cloud Q via rigid rotation."""
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q
    H = np.dot(P_c.T, Q_c)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    return np.dot(P_c, R) + centroid_Q

# --- 4. MAIN EXECUTION ---
def run_full_experiment():
    # A. TRAIN
    dataset = GlassDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # MOVE MODEL TO DEVICE IMMEDIATELY
    model = GlassTimeReversal().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("\n--- Phase 3: Shift-Invariant Training ---")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_disp = model(batch)
            pred_initial = batch.pos - pred_disp
            loss = centered_mse_loss(pred_initial, batch.y, batch.batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.6f}")

    # B. VISUALIZE (With the SAME model instance)
    print("\n--- Phase 4: Kabsch Visualization ---")
    model.eval()
    
    # Check 3 random samples
    indices = [0, 1, 2]
    fig = plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx].to(device)
        with torch.no_grad():
            pred_disp = model(sample)
            reconstructed = sample.pos - pred_disp
            
        glass = sample.pos.cpu().numpy()
        truth = sample.y.cpu().numpy()
        recon = reconstructed.cpu().numpy()
        
        # Apply Kabsch
        recon_aligned = kabsch_align(recon, truth)
        
        # Plot
        ax1 = fig.add_subplot(3, 3, i*3 + 1, projection='3d')
        ax1.scatter(glass[:,0], glass[:,1], glass[:,2], c='blue', alpha=0.1)
        if i==0: ax1.set_title("Scrambled Glass")
        ax1.set_axis_off()
        
        ax2 = fig.add_subplot(3, 3, i*3 + 2, projection='3d')
        ax2.scatter(recon_aligned[:,0], recon_aligned[:,1], recon_aligned[:,2], c='red', alpha=0.6, s=20)
        if i==0: ax2.set_title("AI Reconstruction (Aligned)")
        ax2.set_axis_off()
        
        ax3 = fig.add_subplot(3, 3, i*3 + 3, projection='3d')
        ax3.scatter(truth[:,0], truth[:,1], truth[:,2], c='green', alpha=0.6, s=20)
        if i==0: ax3.set_title("Ground Truth")
        ax3.set_axis_off()
        
    plt.tight_layout()
    plt.savefig("time_reversal_checkmate.png")
    print("Success. Open 'time_reversal_checkmate.png' to see the reconstruction.")

if __name__ == "__main__":
    run_full_experiment()