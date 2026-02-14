import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GlassTimeReversal(torch.nn.Module):
    def __init__(self):
        super(GlassTimeReversal, self).__init__()
        
        # --- INPUT ENCODER ---
        # Input Dim = 6:
        #   3 dimensions for Hessian Eigenvalues (Local stiffness)
        #   3 dimensions for Current Coordinates (Normalized x,y,z)
        self.encoder = nn.Linear(6, 128) 
        
        # --- PROCESSOR (The "Brain") ---
        # Using GATv2 to allow the model to "attend" to force chains
        # We increase hidden dim to 128 to hold more spatial info
        self.gat1 = GATv2Conv(128, 128, heads=4, edge_dim=1, concat=False)
        self.gat2 = GATv2Conv(128, 256, heads=4, edge_dim=1, concat=False)
        self.gat3 = GATv2Conv(256, 128, heads=4, edge_dim=1, concat=False)

        # --- DECODER (The "Un-muddler") ---
        # NO POOLING LAYER. We decode per-node.
        self.node_decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # FINAL OUTPUT: 3D Displacement Vector (dx, dy, dz)
            nn.Linear(32, 3) 
        )

    def forward(self, data):
        # Unpack data
        # x: Hessian Eigenvalues [N, 3]
        # pos: Current normalized positions [N, 3]
        # edge_attr: Bond lengths [E, 1]
        
        x, pos, edge_index, edge_attr = data.x, data.pos, data.edge_index, data.edge_attr

        # 1. Fuse Geometry and Topology
        # We concatenate the physical position with the Hessian signature
        # This tells the model: "I am HERE, and I am stressed like THIS."
        x = torch.cat([x, pos], dim=1) # Shape: [N, 6]
        
        # 2. Encode
        x = F.relu(self.encoder(x))
        
        # 3. Message Passing (Process the Force Chains)
        # Residual connections (x + ...) help preserve position info deeper in the network
        x_in = x
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x) + x_in # Residual skip
        
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        x = self.gat3(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # 4. Decode (No Pooling!)
        # x is still [N, 128]. We treat each node independently now.
        displacement = self.node_decoder(x) # Output: [N, 3]
        
        return displacement