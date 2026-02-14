"""
TOPOLOGY-ONLY VALIDATION SUITE
===============================
This script tests what information the GNN actually needs.

We'll run 5 experiments with increasingly stripped data:
1. BASELINE: Full features + positions + distances (your original)
2. RANDOM FEATURES: Keep graph structure, randomize node features
3. TOPOLOGY ONLY: Only connectivity (node degree), no distances
4. ADVANCED TOPO: Graph-theoretic features (clustering, centrality)
5. PERSISTENT HOMOLOGY: Pure topological invariants

If accuracy stays high even in (3) or (4), you've found geometric signal.
If it drops below 65%, the signal was mostly in features/distances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import networkx as nx
from collections import defaultdict

# ============================================================================
# PART 1: GRAPH BUILDERS (Different Information Levels)
# ============================================================================

def build_full_graph(item, label, scaler=None):
    """Original: Full features + positions + edge distances"""
    raw_feats = np.nan_to_num(item['features'], posinf=100.0, neginf=-100.0)
    
    if scaler is not None:
        norm_feats = scaler.transform(raw_feats)
    else:
        norm_feats = raw_feats
    
    pos = torch.tensor(item['positions'], dtype=torch.float)
    x = torch.tensor(norm_feats, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.float)
    
    # Build edges with distance cutoff
    cutoff = 2.5
    dist_matrix = torch.cdist(pos, pos)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    row, col = edge_index
    edge_attr = dist_matrix[row, col].unsqueeze(1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)


def build_random_features_graph(item, label):
    """Test 2: Random features, keep graph structure and distances"""
    pos = torch.tensor(item['positions'], dtype=torch.float)
    n_nodes = len(pos)
    
    # RANDOM FEATURES (break feature-geometry correlation)
    x = torch.randn(n_nodes, 3)  # Same dimension as original features
    y = torch.tensor([label], dtype=torch.float)
    
    cutoff = 2.5
    dist_matrix = torch.cdist(pos, pos)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    row, col = edge_index
    edge_attr = dist_matrix[row, col].unsqueeze(1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_topology_only_graph(item, label):
    """Test 3: ONLY connectivity - no distances, no meaningful features"""
    pos = torch.tensor(item['positions'], dtype=torch.float)
    n_nodes = len(pos)
    
    cutoff = 2.5
    dist_matrix = torch.cdist(pos, pos)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    # Node features: ONLY the degree (how many neighbors)
    degrees = mask.sum(dim=1).float().unsqueeze(1)
    x = degrees
    
    # NO edge attributes (no distances!)
    y = torch.tensor([label], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=None, y=y)


def compute_graph_features(pos, edge_index, n_nodes):
    """Compute graph-theoretic features for each node"""
    # Convert to NetworkX for analysis
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)
    
    features = []
    
    # Degree
    degrees = dict(G.degree())
    
    # Clustering coefficient
    clustering = nx.clustering(G)
    
    # Betweenness centrality (expensive, use approximation)
    try:
        betweenness = nx.betweenness_centrality(G, k=min(50, n_nodes))
    except:
        betweenness = {i: 0 for i in range(n_nodes)}
    
    # Closeness centrality
    try:
        closeness = nx.closeness_centrality(G)
    except:
        closeness = {i: 0 for i in range(n_nodes)}
    
    # Local efficiency
    local_eff = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            subgraph = G.subgraph(neighbors)
            try:
                local_eff[node] = nx.global_efficiency(subgraph)
            except:
                local_eff[node] = 0
        else:
            local_eff[node] = 0
    
    # Assemble features per node
    for i in range(n_nodes):
        features.append([
            degrees.get(i, 0),
            clustering.get(i, 0),
            betweenness.get(i, 0),
            closeness.get(i, 0),
            local_eff.get(i, 0)
        ])
    
    return np.array(features)


def build_advanced_topo_graph(item, label):
    """Test 4: Graph-theoretic features (no positions, no distances)"""
    pos = torch.tensor(item['positions'], dtype=torch.float)
    n_nodes = len(pos)
    
    cutoff = 2.5
    dist_matrix = torch.cdist(pos, pos)
    mask = (dist_matrix < cutoff) & (dist_matrix > 0)
    edge_index = mask.nonzero().t()
    
    # Compute topological features
    topo_feats = compute_graph_features(pos, edge_index, n_nodes)
    
    x = torch.tensor(topo_feats, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=None, y=y)


def build_persistent_homology_graph(item, label):
    """Test 5: Persistent homology features (PURE topology)"""
    try:
        from scipy.spatial.distance import pdist, squareform
        pos = item['positions']
        n_nodes = len(pos)
        
        # Build distance matrix
        dist_matrix = squareform(pdist(pos))
        
        # Compute persistence features (simplified version)
        # We'll use distance distribution and nearest neighbor stats
        cutoff = 2.5
        
        # For each point, get distances to neighbors
        neighbor_dists = []
        for i in range(n_nodes):
            dists = dist_matrix[i]
            neighbors = dists[dists < cutoff]
            if len(neighbors) > 1:
                neighbor_dists.extend(neighbors[neighbors > 0])
        
        # Statistical features of the point cloud
        if len(neighbor_dists) > 0:
            global_feats = [
                np.mean(neighbor_dists),
                np.std(neighbor_dists),
                np.percentile(neighbor_dists, 25),
                np.percentile(neighbor_dists, 75),
                len(neighbor_dists) / n_nodes  # Average coordination
            ]
        else:
            global_feats = [0, 0, 0, 0, 0]
        
        # Replicate to all nodes (graph-level features)
        x = torch.tensor([global_feats] * n_nodes, dtype=torch.float)
        
        # Build edges
        pos_tensor = torch.tensor(pos, dtype=torch.float)
        dist_matrix_torch = torch.cdist(pos_tensor, pos_tensor)
        mask = (dist_matrix_torch < cutoff) & (dist_matrix_torch > 0)
        edge_index = mask.nonzero().t()
        
        y = torch.tensor([label], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=None, y=y)
    except Exception as e:
        print(f"PH computation failed: {e}")
        return build_topology_only_graph(item, label)


# ============================================================================
# PART 2: MODELS (Adapted for different input types)
# ============================================================================

class FlexibleGNN(torch.nn.Module):
    """GNN that adapts to different feature dimensions and edge attributes"""
    def __init__(self, input_dim, use_edge_attr=True):
        super(FlexibleGNN, self).__init__()
        
        self.use_edge_attr = use_edge_attr
        self.encoder = nn.Linear(input_dim, 64)
        
        edge_dim = 1 if use_edge_attr else None
        
        self.gat1 = GATv2Conv(64, 64, heads=4, edge_dim=edge_dim, concat=True)
        self.gat2 = GATv2Conv(256, 64, heads=4, edge_dim=edge_dim, concat=True)
        
        self.post_mp = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(64, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if self.use_edge_attr else None
        
        x = F.relu(self.encoder(x))
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        x = global_mean_pool(x, data.batch)
        x = self.post_mp(x)
        
        return self.classifier(x)


# ============================================================================
# PART 3: TRAINING FUNCTION
# ============================================================================

def train_and_evaluate(train_loader, test_loader, input_dim, use_edge_attr, 
                       experiment_name, epochs=50, verbose=True):
    """Train a GNN and return test accuracy"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlexibleGNN(input_dim, use_edge_attr).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"{'='*60}")
        print(f"Training on {device}...")
    
    best_acc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out.view(-1), batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    pred_logits = model(batch)
                    probs = torch.sigmoid(pred_logits)
                    predicted_label = (probs > 0.5).float()
                    
                    all_preds.extend(predicted_label.view(-1).cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds) * 100
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}: Loss = {total_loss:.4f} | Test Acc = {acc:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
    
    if verbose:
        print(f"\nBest Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    
    return best_acc


# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("TOPOLOGY-ONLY VALIDATION SUITE")
    print("="*70)
    print("\nLoading data...")
    
    with open('glass_challenge_data.pkl', 'rb') as f:
        raw_data, raw_labels = pickle.load(f)
    
    print(f"Loaded {len(raw_data)} samples")
    
    # Prepare global scaler for baseline
    from sklearn.preprocessing import StandardScaler
    all_features = []
    for item in raw_data:
        feats = np.nan_to_num(item['features'], posinf=100.0, neginf=-100.0)
        all_features.append(feats)
    all_features_stacked = np.vstack(all_features)
    scaler = StandardScaler()
    scaler.fit(all_features_stacked)
    
    # Results storage
    results = {}
    
    # ========================================================================
    # EXPERIMENT 1: BASELINE (Full Information)
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: BASELINE - Full features + positions + distances")
    print("="*70)
    
    data_list = []
    for i, item in enumerate(raw_data):
        data_list.append(build_full_graph(item, raw_labels[i], scaler))
    
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    results['baseline'] = train_and_evaluate(
        train_loader, test_loader, 
        input_dim=3, 
        use_edge_attr=True,
        experiment_name="BASELINE",
        epochs=50
    )
    
    # ========================================================================
    # EXPERIMENT 2: RANDOM FEATURES
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: Random node features (keeps graph structure)")
    print("="*70)
    
    data_list = []
    for i, item in enumerate(raw_data):
        data_list.append(build_random_features_graph(item, raw_labels[i]))
    
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    results['random_features'] = train_and_evaluate(
        train_loader, test_loader,
        input_dim=3,
        use_edge_attr=True,
        experiment_name="RANDOM FEATURES",
        epochs=50
    )
    
    # ========================================================================
    # EXPERIMENT 3: TOPOLOGY ONLY (Node Degree)
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: TOPOLOGY ONLY - Only node degree, no distances")
    print("="*70)
    
    data_list = []
    for i, item in enumerate(raw_data):
        data_list.append(build_topology_only_graph(item, raw_labels[i]))
    
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    results['topology_only'] = train_and_evaluate(
        train_loader, test_loader,
        input_dim=1,
        use_edge_attr=False,
        experiment_name="TOPOLOGY ONLY",
        epochs=50
    )
    
    # ========================================================================
    # EXPERIMENT 4: ADVANCED TOPOLOGY
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 4: ADVANCED TOPOLOGY - Graph-theoretic features")
    print("This will take longer (computing centrality measures)...")
    print("="*70)
    
    data_list = []
    for i, item in enumerate(raw_data):
        if i % 100 == 0:
            print(f"  Processing graph {i}/{len(raw_data)}...")
        data_list.append(build_advanced_topo_graph(item, raw_labels[i]))
    
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    results['advanced_topo'] = train_and_evaluate(
        train_loader, test_loader,
        input_dim=5,
        use_edge_attr=False,
        experiment_name="ADVANCED TOPOLOGY",
        epochs=50
    )
    
    # ========================================================================
    # EXPERIMENT 5: PERSISTENT HOMOLOGY
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 5: PERSISTENT HOMOLOGY - Pure topological invariants")
    print("="*70)
    
    data_list = []
    for i, item in enumerate(raw_data):
        data_list.append(build_persistent_homology_graph(item, raw_labels[i]))
    
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    results['persistent_homology'] = train_and_evaluate(
        train_loader, test_loader,
        input_dim=5,
        use_edge_attr=False,
        experiment_name="PERSISTENT HOMOLOGY",
        epochs=50
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Experiment':<30} {'Test Accuracy':>15} {'Interpretation'}")
    print("-" * 70)
    
    interpretations = {
        'baseline': 'Full information',
        'random_features': 'Structure > Features',
        'topology_only': 'Connectivity matters',
        'advanced_topo': 'Graph theory works',
        'persistent_homology': 'Pure topology'
    }
    
    for exp_name, acc in results.items():
        interp = interpretations[exp_name]
        print(f"{exp_name:<30} {acc:>14.2f}% {interp:>20}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("="*70)
    
    baseline_acc = results['baseline']
    topo_acc = results['topology_only']
    advanced_acc = results['advanced_topo']
    random_acc = results['random_features']
    
    print(f"\n1. BASELINE vs RANDOM FEATURES:")
    drop = baseline_acc - random_acc
    print(f"   Accuracy drop: {drop:.2f}%")
    if drop < 10:
        print("   → Features DON'T matter much. Signal is in STRUCTURE.")
    elif drop < 30:
        print("   → Features contribute, but structure is important too.")
    else:
        print("   → Features are CRITICAL. GNN is exploiting feature information.")
    
    print(f"\n2. TOPOLOGY ONLY (degree) performance:")
    if topo_acc > 80:
        print(f"   → {topo_acc:.1f}% with ONLY connectivity! STRONG geometric signal.")
        print("   → The difference is in HOW particles are CONNECTED.")
        print("   → This is the 'force chain' vs 'random' distinction.")
    elif topo_acc > 65:
        print(f"   → {topo_acc:.1f}% is above random. Moderate geometric signal.")
        print("   → Connectivity patterns matter, but not overwhelmingly.")
    else:
        print(f"   → {topo_acc:.1f}% is barely above random (50%).")
        print("   → Signal was mostly in features/distances, not pure topology.")
    
    print(f"\n3. ADVANCED TOPOLOGY performance:")
    if advanced_acc > 85:
        print(f"   → {advanced_acc:.1f}%! Graph-theoretic features are POWERFUL.")
        print("   → Clustering, centrality, etc. capture the glass structure.")
        print("   → THIS IS THE SMOKING GUN for geometric understanding.")
    elif advanced_acc > 70:
        print(f"   → {advanced_acc:.1f}% is good. Some geometric signal exists.")
    else:
        print(f"   → {advanced_acc:.1f}% suggests graph features aren't sufficient.")
    
    print("\n" + "="*70)
    print("FINAL VERDICT:")
    print("="*70)
    
    if topo_acc > 80 or advanced_acc > 85:
        print("✅ YOUR GNN IS LEARNING REAL GEOMETRIC STRUCTURE!")
        print("   The connectivity patterns differ between fast/slow glasses.")
        print("   This validates your 'force chain' hypothesis.")
        print("\n   Next steps:")
        print("   - Visualize which graph features matter most")
        print("   - Test on different cooling rates (not just 2 classes)")
        print("   - Try to predict CONTINUOUS properties (viscosity, etc.)")
        print("\n   Status: PUBLISHABLE with more analysis")
    elif random_acc > 85:
        print("⚠️  SIGNAL IS MOSTLY IN GRAPH STRUCTURE + DISTANCES")
        print("   The GNN works, but it's using metric information (distances).")
        print("   Not quite 'pure topology' but still interesting.")
        print("\n   Status: Good ML work, needs physics interpretation")
    else:
        print("❌ SIGNAL IS IN FEATURES, NOT GEOMETRY")
        print("   The GNN is mostly exploiting the Hessian eigenvalues.")
        print("   The geometric structure is less important than you thought.")
        print("\n   Status: Back to the drawing board")
    
    print("="*70)
    
    # Save results
    import json
    with open('topology_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: topology_validation_results.json")


if __name__ == "__main__":
    main()