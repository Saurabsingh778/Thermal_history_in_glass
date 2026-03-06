"""
GNN Training — KA4096 patches, RTX 4060 8GB
=============================================
THE PURE METRIC GEOMETRY TEST (5D Features)
This version completely removes the Lennard-Jones Hessian to ensure 
100% physical validity on the Kob-Andersen dataset.
"""

import os
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout, LayerNorm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle, json, time
import numpy as np
from tqdm.auto import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = 'glass_challenge_data_KA4096.pkl'
# Changed cache name so it rebuilds the graphs correctly!
GRAPH_CACHE  = 'graphs_KA512_5D_PURE_GEOMETRY.pt'          
BOX_SIZE     = (4096 / 1.2) ** (1/3)

N_SUB        = 512
N_PATCHES    = 3
R_CUT_GRAPH  = 1.5    
# THE FIX: Dropped to 5 features. Hessian is GONE.
NODE_DIM     = 5      # [mean_bond, std_bond, min_bond, max_bond, degree]

BATCH_SIZE   = 4
ACCUM_STEPS  = 4      
EPOCHS       = 30
LR           = 1e-3
N_FOLDS      = 5
USE_AMP      = True
RESULTS_FILE = "cv_results_KA512_5D_PURE_GEOMETRY.json"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION  — rich node features from local geometry
# ═══════════════════════════════════════════════════════════════════════════════

def build_patch(pos_np, hess_np, label, seed):
    """
    pos_np  : (4096, 3) float32 particle positions
    hess_np : IGNORED. We do not use the LJ Hessian for KA!
    """
    rng    = np.random.default_rng(seed)
    center = rng.integers(0, len(pos_np))

    tree    = cKDTree(pos_np, boxsize=[BOX_SIZE]*3)
    _, idxs = tree.query(pos_np[center], k=N_SUB)
    sub_pos  = pos_np[idxs].astype(np.float32)

    sub_tree = cKDTree(sub_pos)
    pairs    = sub_tree.query_pairs(r=R_CUT_GRAPH, output_type='ndarray')
    if len(pairs) == 0:
        return None

    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])

    delta    = sub_pos[src] - sub_pos[dst]
    bond_len = np.linalg.norm(delta, axis=1).astype(np.float32)   

    n  = N_SUB
    bond_sum  = np.zeros(n, dtype=np.float32)
    bond_sum2 = np.zeros(n, dtype=np.float32)
    bond_min  = np.full(n, np.inf, dtype=np.float32)
    bond_max  = np.zeros(n, dtype=np.float32)
    deg       = np.zeros(n, dtype=np.float32)

    np.add.at(bond_sum,  src, bond_len)
    np.add.at(bond_sum2, src, bond_len ** 2)
    np.minimum.at(bond_min, src, bond_len)
    np.maximum.at(bond_max, src, bond_len)
    np.add.at(deg, src, 1.0)

    safe_deg  = np.maximum(deg, 1.0)
    mean_bond = bond_sum  / safe_deg
    std_bond  = np.sqrt(np.maximum(bond_sum2 / safe_deg - mean_bond**2, 0.0))
    bond_min[bond_min == np.inf] = 0.0

    # THE FIX: Stack -> (N_SUB, 5). Pure Metric Geometry.
    node_feats = np.stack(
        [mean_bond, std_bond, bond_min, bond_max, deg / deg.max()],
        axis=1
    ).astype(np.float32)

    mu  = node_feats.mean(axis=0, keepdims=True)
    sig = node_feats.std(axis=0, keepdims=True) + 1e-6
    node_feats = (node_feats - mu) / sig

    return Data(
        x          = torch.from_numpy(node_feats),
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr  = torch.from_numpy(bond_len.reshape(-1, 1)),
        y          = torch.tensor([label], dtype=torch.float),
    )

def load_or_build_graphs(raw_data, labels):
    if os.path.exists(GRAPH_CACHE):
        print(f"Loading cached graphs from {GRAPH_CACHE} …")
        return torch.load(GRAPH_CACHE, weights_only=False)

    print(f"Building {N_PATCHES} patches × {len(raw_data)} samples …")
    t0   = time.time()
    args = [
        (raw_data[i]['positions'], raw_data[i]['features'], labels[i], i * 1000 + p)
        for i in range(len(raw_data))
        for p in range(N_PATCHES)
    ]
    graphs  = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        futs = {pool.submit(build_patch, *a): a for a in args}
        for fut in tqdm(as_completed(futs), total=len(args), desc="Patches"):
            g = fut.result()
            if g is None:
                skipped += 1
            else:
                graphs.append(g)

    torch.save(graphs, GRAPH_CACHE)
    sample_edges = [g.edge_index.shape[1] for g in graphs[:20]]
    print(f"Built {len(graphs)} (skipped {skipped})  {time.time()-t0:.1f}s")
    print(f"Edges — mean={np.mean(sample_edges):.0f}  max={np.max(sample_edges)}")
    return graphs

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL  — 5D node features, same GATv2 architecture as paper
# ═══════════════════════════════════════════════════════════════════════════════

class GlassGAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Linear(NODE_DIM, 64)
        self.conv1   = GATv2Conv(64,  64, heads=4, edge_dim=1, concat=True)   
        self.conv2   = GATv2Conv(256, 64, heads=4, edge_dim=1, concat=True)   
        self.classifier = Sequential(
            Linear(256, 128), LayerNorm(128), ReLU(), Dropout(0.2),
            Linear(128, 64),  ReLU(), Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = F.relu(self.encoder(x))
        h = F.elu(self.conv1(h, edge_index, edge_attr))
        h = F.elu(self.conv2(h, edge_index, edge_attr))
        h = global_mean_pool(h, batch)
        return self.classifier(h)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = batch.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=USE_AMP):
            out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out.squeeze(), batch.y) / ACCUM_STEPS
        scaler.scale(loss).backward()
        total_loss += loss.item() * ACCUM_STEPS * batch.num_graphs
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, preds, labs, total_loss = [], [], [], 0
    crit = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        batch  = batch.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=USE_AMP):
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
            loss   = crit(logits, batch.y)
        total_loss += loss.item() * batch.num_graphs
        p = torch.sigmoid(logits).float().cpu().numpy()
        probs.extend(p)
        preds.extend((p > 0.5).astype(float))
        labs.extend(batch.y.cpu().numpy())
    acc = np.mean(np.array(preds) == np.array(labs))
    auc = roc_auc_score(labs, probs)
    f1  = f1_score(labs, preds, zero_division=0)
    return total_loss / len(loader.dataset), acc, auc, f1, labs, preds

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"Device   : {DEVICE}")
    print(f"Node features: {NODE_DIM}D  [mean_bond, std_bond, min_bond, max_bond, deg]")
    print(f"N_SUB={N_SUB}  r_cut={R_CUT_GRAPH}  patches={N_PATCHES}")
    print(f"batch={BATCH_SIZE}  accum={ACCUM_STEPS}  eff={BATCH_SIZE*ACCUM_STEPS}  AMP={USE_AMP}")

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    with open(DATA_PATH, 'rb') as f:
        raw_data, labels = pickle.load(f)
    print(f"Loaded {len(raw_data)} samples  Fast={labels.count(0)}  Slow={labels.count(1)}")

    dataset    = load_or_build_graphs(raw_data, labels)
    labels_arr = [int(d.y.item()) for d in dataset]

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    all_true, all_pred = [], []

    print(f"\n{'='*55}")
    print(f" STRATIFIED {N_FOLDS}-FOLD CV — KA512 patches (5D features)")
    print(f"{'='*55}")
    t_start = time.time()

    for fold, (tr_idx, te_idx) in enumerate(skf.split(range(len(dataset)), labels_arr)):
        print(f"\n─── FOLD {fold+1}/{N_FOLDS}  train={len(tr_idx)} test={len(te_idx)} ───")

        tr_loader = DataLoader([dataset[i] for i in tr_idx],
                               batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        te_loader = DataLoader([dataset[i] for i in te_idx],
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model     = GlassGAT().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss()

        best_loss, best_state, patience_ctr = float('inf'), None, 0
        EARLY_STOP = 20

        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_epoch(model, tr_loader, optimizer, criterion, scaler)
            val_loss, val_acc, val_auc, val_f1, _, _ = evaluate(model, te_loader)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss    = val_loss
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if epoch % 10 == 0:
                print(f"  ep{epoch:3d}  tr={tr_loss:.4f}  val={val_loss:.4f}"
                      f"  acc={val_acc*100:.1f}%  AUC={val_auc:.3f}"
                      f"  lr={optimizer.param_groups[0]['lr']:.1e}")

            if patience_ctr >= EARLY_STOP:
                print(f"  Early stop at epoch {epoch}")
                break

        model.load_state_dict(best_state)
        _, acc, auc, f1, ft, fp = evaluate(model, te_loader)
        all_true.extend(ft); all_pred.extend(fp)
        fold_results.append({"fold": fold+1, "acc": acc, "auc": auc, "f1": f1})
        print(f"  ✓ Fold {fold+1}  Acc={acc*100:.2f}%  AUC={auc:.4f}  F1={f1:.4f}")

    elapsed = time.time() - t_start
    accs = [r["acc"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    f1s  = [r["f1"]  for r in fold_results]

    print(f"\n{'='*55}")
    print(f" FINAL RESULTS  ({elapsed/60:.1f} min)")
    print(f"{'='*55}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['acc']*100:.2f}%  AUC={r['auc']:.4f}  F1={r['f1']:.4f}")
    print(f"{'─'*55}")
    print(f"  Mean Acc : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  Mean AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Mean F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    cm = confusion_matrix(all_true, all_pred)
    print(f"\nConfusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")
    print(classification_report(all_true, all_pred,
                                target_names=["Fast T=0.64", "Slow T=0.44"]))

    summary = {
        "dataset"    : f"KA4096 patches N={N_SUB} r_cut={R_CUT_GRAPH}",
        "node_features": "mean_bond std_bond min_bond max_bond deg (instance-normed)",
        "n_patches"  : N_PATCHES, "n_sub": N_SUB, "n_folds": N_FOLDS,
        "fold_results": fold_results,
        "mean_acc"   : float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_auc"   : float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
        "mean_f1"    : float(np.mean(f1s)),  "std_f1" : float(np.std(f1s)),
        "confusion_matrix": cm.tolist(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")