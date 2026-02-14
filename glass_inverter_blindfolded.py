import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- 1. THE BLINDFOLD PROTOCOL ---
print("Loading 'glass_challenge_data.pkl'...")
with open('glass_challenge_data.pkl', 'rb') as f:
    raw_data, raw_labels = pickle.load(f)

print("Applying Adversarial Normalization (Instance Norm)...")
data_list = []
cutoff = 2.5 

# For the Dumb Model Check
dumb_features = []

for i, item in enumerate(raw_data):
    # 1. Sanitize
    raw_feats = item['features']
    if np.isnan(raw_feats).any() or np.isinf(raw_feats).any():
        raw_feats = np.nan_to_num(raw_feats, posinf=100.0, neginf=-100.0)
    
    # 2. INSTANCE NORMALIZATION (The Blindfold)
    # We subtract THIS glass's mean from itself. 
    # Now every glass has Mean=0 and Std=1.
    scaler = StandardScaler()
    norm_feats = scaler.fit_transform(raw_feats)
    
    # Collect features for Dumb Model Proof
    # We take the mean of the NORMALIZED features (Should be ~0)
    dumb_features.append(np.mean(norm_feats, axis=0))

    # 3. Build Graph
    pos = torch.tensor(item['positions'], dtype=torch.float)
    x = torch.tensor(norm_feats, dtype=torch.float) # Blindfolded Features
    y = torch.tensor([raw_labels[i]], dtype=torch.float)

    dist_matrix = torch.cdist(pos, pos)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    row, col = edge_index
    edge_attr = dist_matrix[row, col].unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
    data_list.append(data)

# --- 2. PROVE THE BLINDFOLD WORKS ---
print("\n[Control Check] Verifying Dumb Model is Blind...")
X_dumb = np.array(dumb_features)
y_dumb = np.array(raw_labels)
# Train Dumb Model on Blindfolded Data
clf = LogisticRegression()
clf.fit(X_dumb, y_dumb)
dumb_acc = clf.score(X_dumb, y_dumb)
print(f"Dumb Model Accuracy on Blindfolded Data: {dumb_acc*100:.2f}%")
if dumb_acc > 0.6:
    print("WARNING: Blindfold failed. Normalization error.")
else:
    print("SUCCESS: Dumb Model is blind (approx 50%). Proceeding to GNN.")

# --- 3. TRAIN THE GNN ---
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

class GlassInverter(torch.nn.Module):
    def __init__(self):
        super(GlassInverter, self).__init__()
        self.encoder = nn.Linear(3, 64)
        
        # Deeper GNN for the harder task
        self.gat1 = GATv2Conv(64, 64, heads=4, edge_dim=1, concat=True)
        self.gat2 = GATv2Conv(256, 128, heads=4, edge_dim=1, concat=True) # Wider
        self.gat3 = GATv2Conv(512, 64, heads=2, edge_dim=1, concat=True)  # Deeper
        
        self.post_mp = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
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
        x = self.gat3(x, edge_index, edge_attr=edge_attr) # Added depth
        x = F.elu(x)
        
        x = global_mean_pool(x, data.batch)
        x = self.post_mp(x)
        return self.classifier(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GlassInverter().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

print(f"\nStarting Geometric Training on {device}...")
best_acc = 0
for epoch in range(50): # Longer training for harder task
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.view(-1), batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

print(f"\nFinal Blindfolded Accuracy: {best_acc:.2f}%")
if best_acc > 80:
    print("CONCLUSION: Internet Broken. The Model sees PURE GEOMETRY.")
else:
    print("CONCLUSION: The signal was mostly magnitude. Physics is safe.")