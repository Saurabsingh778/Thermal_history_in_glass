import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

# --- 1. MODEL DEFINITION ---
class GlassClassifier(nn.Module):
    def __init__(self): 
        super().__init__()
        self.encoder = nn.Linear(3, 64) # 3D Hessian input
        self.gat1 = GATv2Conv(64, 64, heads=4, edge_dim=1, concat=False)
        self.gat2 = GATv2Conv(64, 64, heads=4, edge_dim=1, concat=False)
        
        self.post_mp = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.encoder(x))
        x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.gat2(x, edge_index, edge_attr=edge_attr))
        x_graph = global_mean_pool(x, batch)
        x_graph = self.post_mp(x_graph)
        return self.classifier(x_graph)

# --- 2. DATA LOADING & GRAPH BUILDING ---
print("Loading 'glass_challenge_data.pkl'...")
with open('glass_challenge_data.pkl', 'rb') as f:
    raw_data, raw_labels = pickle.load(f)

# Fit Global Scaler (Matches your original baseline)
all_features = []
for item in raw_data:
    feats = np.nan_to_num(item['features'], posinf=100.0, neginf=-100.0)
    all_features.append(feats)
scaler = StandardScaler()
scaler.fit(np.vstack(all_features))

dataset = []
cutoff = 2.5 

print("Building Graphs...")
for i, item in enumerate(raw_data):
    raw_feats = np.nan_to_num(item['features'], posinf=100.0, neginf=-100.0)
    norm_feats = scaler.transform(raw_feats)
    
    pos = torch.tensor(item['positions'], dtype=torch.float)
    x = torch.tensor(norm_feats, dtype=torch.float)
    y = torch.tensor([raw_labels[i]], dtype=torch.float)
    
    dist_matrix = torch.cdist(pos, pos)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    row, col = edge_index
    edge_attr = dist_matrix[row, col].unsqueeze(1)
    
    dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

# --- 3. 5-FOLD CV LOGIC ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Starting 5-Fold Stratified CV on {device}...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
labels_arr = np.array(raw_labels)

for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels_arr)), labels_arr)):
    print(f"\n--- FOLD {fold + 1} ---")
    
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=32, shuffle=True)
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=32, shuffle=False)
    
    model = GlassClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    best_metrics = {}
    
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).squeeze()
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch).squeeze()
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).long()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            best_metrics = {
                'acc': acc * 100,
                'auc': roc_auc_score(all_labels, all_probs),
                'f1': f1_score(all_labels, all_preds)
            }
            
    print(f"Fold {fold + 1} Best Acc: {best_metrics['acc']:.2f}% | AUC: {best_metrics['auc']:.4f}")
    fold_results.append(best_metrics)

# Aggregate Results
accs = [res['acc'] for res in fold_results]
aucs = [res['auc'] for res in fold_results]
f1s = [res['f1'] for res in fold_results]

print("\n======================================")
print(f"FINAL LJ BASELINE RESULTS (5-Fold CV)")
print(f"Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
print(f"AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print("======================================")