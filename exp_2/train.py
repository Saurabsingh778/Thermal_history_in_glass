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
EPOCHS = 150        # Increased epochs (shape learning takes longer)
BOX_SIZE = 15.0   
RADIUS = 2.5     

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- HELPER: RADIUS GRAPH ---
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

# --- 2. THE MODEL (Unchanged) ---
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

# --- 3. THE FIX: SHIFT-INVARIANT LOSS ---
def centered_mse_loss(pred_pos, target_pos, batch_indices):
    """
    Computes MSE after aligning the Centers of Mass (COM) of prediction and target.
    """
    # 1. Compute COM for each batch
    # scatter_mean computes the mean position for each graph in the batch
    from torch_geometric.utils import scatter
    
    pred_com = scatter(pred_pos, batch_indices, dim=0, reduce='mean')
    target_com = scatter(target_pos, batch_indices, dim=0, reduce='mean')
    
    # 2. Shift both clouds to be centered at (0,0,0)
    # We map the COM back to each node using the batch_indices
    pred_centered = pred_pos - pred_com[batch_indices]
    target_centered = target_pos - target_com[batch_indices]
    
    # 3. Compute MSE on the SHAPE, not the absolute position
    # Note: We skip PBC wrapping here for simplicity because after centering,
    # if the shape is recovered, the distances should be small.
    # If your "S" wraps around the edge, this simple centering might jitter, 
    # but for small displacements it is robust.
    return F.mse_loss(pred_centered, target_centered)

# --- 4. TRAINING LOOP ---
def train():
    dataset = GlassDataset("time_reversal_data.pkl")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = GlassTimeReversal().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("\nStarting Shift-Invariant Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Predict Displacement
            pred_disp = model(batch)
            
            # Reconstruct
            pred_initial = batch.pos - pred_disp
            
            # --- USE NEW LOSS ---
            loss = centered_mse_loss(pred_initial, batch.y, batch.batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

    # --- 5. VISUALIZATION (With Alignment) ---
    print("\nVisualizing...")
    model.eval()
    test_sample = dataset[0].to(device)
    with torch.no_grad():
        pred_disp = model(test_sample)
        reconstructed = test_sample.pos - pred_disp
        
        # Manually Center for Visualization
        recon_com = reconstructed.mean(dim=0)
        truth_com = test_sample.y.mean(dim=0)
        
        reconstructed_centered = reconstructed - recon_com + 0.5
        truth_centered = test_sample.y - truth_com + 0.5

    glass = test_sample.pos.cpu().numpy()
    truth = truth_centered.cpu().numpy()
    recon = reconstructed_centered.cpu().numpy()
    
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(glass[:,0], glass[:,1], glass[:,2], c='blue', alpha=0.3)
    ax1.set_title("Input: Disordered Glass")
    
    ax2 = fig.add_subplot(132, projection='3d')
    # Use red for AI, but larger points to see structure
    ax2.scatter(recon[:,0], recon[:,1], recon[:,2], c='red', alpha=0.6, s=20) 
    ax2.set_title("AI Reconstruction (Aligned)")
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(truth[:,0], truth[:,1], truth[:,2], c='green', alpha=0.6, s=20)
    ax3.set_title("Ground Truth ('S' Shape)")
    
    plt.savefig("time_reversal_aligned.png")
    print("Saved 'time_reversal_aligned.png'")

if __name__ == "__main__":
    train()