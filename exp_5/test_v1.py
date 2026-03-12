"""
Phase 2: Structural Fictive Temperature via Latent Space Thermometry
=====================================================================
ZERO torch_geometric dependency — GATv2Conv and batching implemented
in pure PyTorch. Works on any PyTorch version including 2.9+.

Input:  fictive_temp_dataset.pkl
Output: phase2_results/  (same structure as before)
"""

# ── 0. IMPORTS — no torch_geometric anywhere ─────────────────────────────────
import os, pickle, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
print("All imports OK — running pure PyTorch, no torch_geometric needed.")

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
CFG = {
    "data_path"   : "/kaggle/input/datasets/saurabingh/fecttiv-dataset/fictive_temp_dataset.pkl",
    "out_dir"     : "phase2_results",
    "box_size"    : 15.0,
    "rc"          : 2.5,
    "hidden_dim"  : 64,
    "heads"       : 4,
    "dropout"     : 0.2,
    "input_dim"   : 3,
    "lr"          : 1e-3,
    "batch_size"  : 32,
    "max_epochs"  : 150,
    "patience"    : 20,
    "n_folds"     : 5,
    "seed"        : 42,
    "n_components": 20,
}

os.makedirs(CFG["out_dir"], exist_ok=True)
os.makedirs(os.path.join(CFG["out_dir"], "figures"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])


# ── 2. GRAPH DATA STRUCTURE (replaces PyG Data) ──────────────────────────────
class GlassGraph:
    """Minimal graph container — same fields as PyG Data."""
    def __init__(self, x, edge_index, edge_attr, y, u_per_n, log10_gamma):
        self.x           = x            # (N, 3)  float32
        self.edge_index  = edge_index   # (2, E)  int64
        self.edge_attr   = edge_attr    # (E, 1)  float32
        self.y           = y            # scalar  float32
        self.u_per_n     = u_per_n      # scalar  float32
        self.log10_gamma = log10_gamma  # scalar  float32
        self.num_nodes   = x.shape[0]


def build_graph(sample, box_size, rc):
    R = sample["positions"].astype(np.float32)
    x = sample["features"].astype(np.float32)
    y = np.float32(sample["log10_gamma"])
    N = R.shape[0]

    diff = R[:, None, :] - R[None, :, :]
    diff -= box_size * np.round(diff / box_size)
    dist = np.linalg.norm(diff, axis=-1)

    src, dst   = np.where((dist < rc) & (dist > 0))
    edge_index = np.stack([src, dst], axis=0)
    edge_attr  = dist[src, dst].reshape(-1, 1)

    return GlassGraph(
        x           = torch.tensor(x),
        edge_index  = torch.tensor(edge_index, dtype=torch.long),
        edge_attr   = torch.tensor(edge_attr),
        y           = torch.tensor(y),
        u_per_n     = torch.tensor(np.float32(sample["u_per_n"])),
        log10_gamma = torch.tensor(y),
    )


# ── 3. PURE-PYTORCH BATCH COLLATOR ───────────────────────────────────────────
def collate_graphs(graph_list):
    """
    Packs a list of GlassGraph into a single batch dict.
    Offsets edge indices so all graphs share one node index space.
    Returns a plain dict — no PyG dependency.
    """
    xs, eis, eas, ys, us, gs, batch_vec = [], [], [], [], [], [], []
    node_offset = 0

    for gid, g in enumerate(graph_list):
        n = g.num_nodes
        xs.append(g.x)
        eis.append(g.edge_index + node_offset)
        eas.append(g.edge_attr)
        ys.append(g.y.unsqueeze(0))
        us.append(g.u_per_n.unsqueeze(0))
        gs.append(g.log10_gamma.unsqueeze(0))
        batch_vec.append(torch.full((n,), gid, dtype=torch.long))
        node_offset += n

    return {
        "x"          : torch.cat(xs,  dim=0),
        "edge_index" : torch.cat(eis, dim=1),
        "edge_attr"  : torch.cat(eas, dim=0),
        "y"          : torch.stack(ys).unsqueeze(1),   # (B, 1, 1)
        "u_per_n"    : torch.stack(us),
        "log10_gamma": torch.stack(gs),
        "batch"      : torch.cat(batch_vec, dim=0),
        "num_graphs" : len(graph_list),
    }


class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]

def make_loader(graphs, batch_size, shuffle):
    return DataLoader(
        GraphDataset(graphs),
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = collate_graphs,
    )


# ── 4. PURE-PYTORCH GATv2Conv ────────────────────────────────────────────────
class GATv2Conv(nn.Module):
    """
    GATv2 message-passing layer (Brody et al. 2021) — pure PyTorch.
    Identical in spirit to PyG's GATv2Conv.
    """
    def __init__(self, in_channels, out_channels, heads=1,
                 edge_dim=None, concat=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.heads        = heads
        self.concat       = concat

        self.W_src  = nn.Linear(in_channels,  heads * out_channels, bias=False)
        self.W_dst  = nn.Linear(in_channels,  heads * out_channels, bias=False)
        self.W_edge = (nn.Linear(edge_dim, heads * out_channels, bias=False)
                       if edge_dim is not None else None)
        self.a      = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias   = nn.Parameter(torch.zeros(
            heads * out_channels if concat else out_channels))
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index, edge_attr=None):
        src_idx, dst_idx = edge_index[0], edge_index[1]
        N = x.size(0)
        H, C = self.heads, self.out_channels

        # Linear projections → (N, H, C)
        h_src = self.W_src(x).view(N, H, C)
        h_dst = self.W_dst(x).view(N, H, C)

        # Messages: e(i→j) = h_src[i] + h_dst[j]  (+ edge if given)
        msg = h_src[src_idx] + h_dst[dst_idx]        # (E, H, C)
        if self.W_edge is not None and edge_attr is not None:
            msg = msg + self.W_edge(edge_attr).view(-1, H, C)

        # Attention score: LeakyReLU then dot with a
        e = F.leaky_relu(msg, 0.2)                    # (E, H, C)
        e = (e * self.a).sum(dim=-1, keepdim=True)    # (E, H, 1)

        # Softmax over neighbourhood of each destination node
        # Use scatter softmax: exp(e) / sum_neighbours exp(e)
        e_exp = torch.exp(e - e.max())                # numerical stability
        denom = torch.zeros(N, H, 1, device=x.device)
        denom.scatter_add_(0, dst_idx.view(-1,1,1).expand_as(e_exp), e_exp)
        denom = denom[dst_idx].clamp(min=1e-6)
        alpha = e_exp / denom                          # (E, H, 1)

        # Weighted aggregation
        out = torch.zeros(N, H, C, device=x.device)
        weighted = (h_src[src_idx] * alpha)            # (E, H, C)
        out.scatter_add_(0,
            dst_idx.view(-1,1,1).expand_as(weighted), weighted)

        if self.concat:
            out = out.view(N, H * C) + self.bias
        else:
            out = out.mean(dim=1) + self.bias
        return out


# ── 5. GLOBAL MEAN POOL (replaces PyG's global_mean_pool) ────────────────────
def global_mean_pool(x, batch, num_graphs):
    """Average node features per graph."""
    out = torch.zeros(num_graphs, x.size(1), device=x.device)
    count = torch.zeros(num_graphs, 1, device=x.device)
    idx   = batch.unsqueeze(1).expand_as(x)
    out.scatter_add_(0, idx, x)
    count.scatter_add_(0, batch.unsqueeze(1),
                       torch.ones(batch.size(0), 1, device=x.device))
    return out / count.clamp(min=1)


# ── 6. MODEL ─────────────────────────────────────────────────────────────────
class GATv2Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, heads=4, dropout=0.2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gat1    = GATv2Conv(hidden_dim, hidden_dim,
                                 heads=heads, edge_dim=1, concat=True)
        self.gat2    = GATv2Conv(hidden_dim * heads, hidden_dim,
                                 heads=heads, edge_dim=1, concat=True)
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim * heads, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
        )
        self.regressor = nn.Linear(64, 1)

    def forward(self, batch_dict, return_latent=False):
        x   = batch_dict["x"].to(DEVICE)
        ei  = batch_dict["edge_index"].to(DEVICE)
        ea  = batch_dict["edge_attr"].to(DEVICE)
        bv  = batch_dict["batch"].to(DEVICE)
        ng  = batch_dict["num_graphs"]

        h       = F.relu(self.encoder(x))
        h       = F.relu(self.gat1(h, ei, ea))
        h       = F.relu(self.gat2(h, ei, ea))
        h_graph = global_mean_pool(h, bv, ng)
        latent  = self.post_mp(h_graph)
        out     = self.regressor(latent)

        if return_latent:
            return out, latent
        return out


# ── 7. TRAINING UTILITIES ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    n_total    = 0
    for batch in loader:
        y    = batch["y"].squeeze().to(DEVICE).float()
        # y shape fix: ensure (B,) or (B,1)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        optimizer.zero_grad()
        pred = model(batch).squeeze()
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        loss = F.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch["num_graphs"]
        n_total    += batch["num_graphs"]
    return total_loss / n_total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    n_total    = 0
    preds, trues = [], []
    for batch in loader:
        y    = batch["y"].squeeze().to(DEVICE).float()
        if y.dim() == 0:
            y = y.unsqueeze(0)
        pred = model(batch).squeeze()
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        loss = F.mse_loss(pred, y)
        total_loss += loss.item() * batch["num_graphs"]
        n_total    += batch["num_graphs"]
        preds.extend(pred.cpu().numpy().flatten())
        trues.extend(y.cpu().numpy().flatten())
    return total_loss / n_total, np.array(preds), np.array(trues)


@torch.no_grad()
def extract_latent(model, loader):
    model.eval()
    latents, preds, trues, energies = [], [], [], []
    for batch in loader:
        pred, lat = model(batch, return_latent=True)
        y = batch["y"].squeeze().float()
        if y.dim() == 0:
            y = y.unsqueeze(0)
        latents.append(lat.cpu().numpy())
        preds.extend(pred.cpu().numpy().flatten())
        trues.extend(y.numpy().flatten())
        energies.extend(batch["u_per_n"].squeeze().numpy().flatten()
                        if batch["u_per_n"].squeeze().dim() > 0
                        else [batch["u_per_n"].item()])
    return (np.vstack(latents),
            np.array(preds),
            np.array(trues),
            np.array(energies))


# ── 8. LOAD DATA ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset ...")
with open(CFG["data_path"], "rb") as f:
    raw = pickle.load(f)
print(f"      {len(raw)} samples loaded")
print(f"      Cooling rates (log10): "
      f"{sorted(set(round(s['log10_gamma'],3) for s in raw))}")

print("[2/5] Building graphs ...")
t0 = time.time()
graphs, nan_count = [], 0
for s in raw:
    g = build_graph(s, CFG["box_size"], CFG["rc"])
    if torch.isnan(g.x).any() or torch.isnan(g.edge_attr).any():
        nan_count += 1
        continue
    graphs.append(g)
print(f"      {len(graphs)} valid graphs  ({nan_count} NaN dropped)"
      f"  [{time.time()-t0:.1f}s]")

labels       = np.array([g.log10_gamma.item() for g in graphs])
energies     = np.array([g.u_per_n.item()     for g in graphs])
unique_rates = np.unique(labels)
print(f"      Label range: [{labels.min():.2f}, {labels.max():.2f}]")


# ── 9. 5-FOLD CROSS-VALIDATION ───────────────────────────────────────────────
print("\n[3/5] 5-Fold Cross-Validation ...")

rate_bins = np.digitize(
    labels,
    bins=np.linspace(labels.min()-0.01, labels.max()+0.01, 9))
kf = KFold(n_splits=CFG["n_folds"], shuffle=True,
           random_state=CFG["seed"])

fold_metrics   = []
all_val_preds  = np.zeros(len(graphs))
all_val_trues  = np.zeros(len(graphs))
all_val_energy = np.zeros(len(graphs))
all_latents    = np.zeros((len(graphs), 64))
indices        = np.arange(len(graphs))

for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(indices, rate_bins)):

    print(f"\n  ── Fold {fold_idx+1}/{CFG['n_folds']} "
          f"(train={len(train_idx)}, val={len(val_idx)}) ──")

    train_loader = make_loader([graphs[i] for i in train_idx],
                               CFG["batch_size"], shuffle=True)
    val_loader   = make_loader([graphs[i] for i in val_idx],
                               CFG["batch_size"], shuffle=False)
    full_loader  = make_loader(graphs, CFG["batch_size"], shuffle=False)

    model = GATv2Regressor(
        input_dim  = CFG["input_dim"],
        hidden_dim = CFG["hidden_dim"],
        heads      = CFG["heads"],
        dropout    = CFG["dropout"],
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=CFG["lr"])
    scheduler = ReduceLROnPlateau(optimizer, patience=10,
                                  factor=0.5, min_lr=1e-5)

    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 1
    patience_ctr  = 0
    train_curve, val_curve = [], []

    for epoch in range(1, CFG["max_epochs"] + 1):
        tr_loss              = train_epoch(model, train_loader, optimizer)
        vl_loss, vl_p, vl_t = eval_epoch(model, val_loader)
        scheduler.step(vl_loss)
        train_curve.append(tr_loss)
        val_curve.append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            best_epoch    = epoch
            patience_ctr  = 0
        else:
            patience_ctr += 1

        if epoch % 20 == 0:
            r2 = r2_score(vl_t, vl_p)
            print(f"    Epoch {epoch:>3} | Train {tr_loss:.4f} | "
                  f"Val {vl_loss:.4f} | R² {r2:.4f}")

        if patience_ctr >= CFG["patience"]:
            print(f"    Early stop epoch {epoch} "
                  f"(best={best_epoch}, MSE={best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    _, vp, vt = eval_epoch(model, val_loader)
    fold_r2   = r2_score(vt, vp)
    fold_mae  = mean_absolute_error(vt, vp)
    fold_rmse = float(np.sqrt(np.mean((vp-vt)**2)))
    print(f"  Fold {fold_idx+1} → R²={fold_r2:.4f}  "
          f"MAE={fold_mae:.4f}  RMSE={fold_rmse:.4f}")

    fold_metrics.append({
        "fold": fold_idx+1, "best_epoch": best_epoch,
        "val_mse": best_val_loss, "r2": fold_r2,
        "mae": fold_mae, "rmse": fold_rmse,
        "train_curve": train_curve, "val_curve": val_curve,
    })

    all_val_preds[val_idx]  = vp
    all_val_trues[val_idx]  = vt
    all_val_energy[val_idx] = energies[val_idx]

    lat, _, _, _ = extract_latent(model, full_loader)
    all_latents += lat / CFG["n_folds"]


# ── 10. SUMMARY ──────────────────────────────────────────────────────────────
r2_vals   = [m["r2"]   for m in fold_metrics]
mae_vals  = [m["mae"]  for m in fold_metrics]
rmse_vals = [m["rmse"] for m in fold_metrics]

print("\n" + "="*60)
print("  5-FOLD CV SUMMARY")
print("="*60)
print(f"  R²   : {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
print(f"  MAE  : {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}  [log10]")
print(f"  RMSE : {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}  [log10]")
print("="*60)

print("\n  Per-rate breakdown:")
print(f"  {'log10(Γ)':>10}  {'N':>4}  {'Mean pred':>10}  {'MAE':>8}")
for rate in unique_rates:
    mask  = all_val_trues == rate
    mae_r = mean_absolute_error(all_val_trues[mask], all_val_preds[mask])
    print(f"  {rate:>10.3f}  {mask.sum():>4}  "
          f"{all_val_preds[mask].mean():>10.4f}  {mae_r:>8.4f}")


# ── 11. PCA ───────────────────────────────────────────────────────────────────
print("\n[4/5] PCA of latent space ...")
scaler         = StandardScaler()
latents_scaled = scaler.fit_transform(all_latents)
pca            = PCA(n_components=CFG["n_components"],
                     random_state=CFG["seed"])
projections    = pca.fit_transform(latents_scaled)
explained      = pca.explained_variance_ratio_

print(f"  Var: PC1={explained[0]:.3f}  PC2={explained[1]:.3f}  "
      f"PC3={explained[2]:.3f}  cum(5)={explained[:5].sum():.3f}")

pc1 = projections[:, 0]
r2_pc1_gamma  = r2_score(labels,
    LinearRegression().fit(pc1.reshape(-1,1), labels)
                      .predict(pc1.reshape(-1,1)))
r2_pc1_energy = r2_score(energies,
    LinearRegression().fit(pc1.reshape(-1,1), energies)
                      .predict(pc1.reshape(-1,1)))

print(f"\n  TRIPLE CORRELATION:")
print(f"  PC1 ↔ log10(Γ) : R² = {r2_pc1_gamma:.4f}")
print(f"  PC1 ↔ U/N       : R² = {r2_pc1_energy:.4f}")
print(f"  (Target: both > 0.90)")


# ── 12. FIGURES ───────────────────────────────────────────────────────────────
print("\n[5/5] Generating figures ...")
FIG = os.path.join(CFG["out_dir"], "figures")

# Fig 1 — Regression scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("GATv2 Regression: Predicted vs True log₁₀(Γ)", fontsize=14)
ax = axes[0]
sc = ax.scatter(all_val_trues, all_val_preds, c=all_val_trues,
                cmap="plasma", alpha=0.55, s=18, linewidths=0)
lims = [all_val_trues.min()-0.1, all_val_trues.max()+0.1]
ax.plot(lims, lims, "k--", lw=1.2, label="y = x")
ax.set_xlabel("True log₁₀(Γ)"); ax.set_ylabel("Predicted log₁₀(Γ)")
ax.set_title(f"R² = {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
plt.colorbar(sc, ax=ax, label="log₁₀(Γ)"); ax.legend()
ax2 = axes[1]
ax2.scatter(all_val_trues, all_val_preds-all_val_trues, c=all_val_trues,
            cmap="plasma", alpha=0.55, s=18, linewidths=0)
ax2.axhline(0, color="k", linestyle="--", lw=1.2)
ax2.set_xlabel("True log₁₀(Γ)"); ax2.set_ylabel("Residual")
ax2.set_title("Residuals")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_regression_scatter.png"), dpi=150)
plt.close()

# Fig 2 — PCA latent space
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Latent Space PCA — Structural Fictive Temperature Manifold")
ax = axes[0]
sc = ax.scatter(projections[:,0], projections[:,1], c=labels,
                cmap="plasma", alpha=0.6, s=16, linewidths=0)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
ax.set_title("Coloured by log₁₀(Γ)")
plt.colorbar(sc, ax=ax, label="log₁₀(Γ)")
ax = axes[1]
sc = ax.scatter(projections[:,0], projections[:,1], c=energies,
                cmap="coolwarm", alpha=0.6, s=16, linewidths=0)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
ax.set_title("Coloured by U/N")
plt.colorbar(sc, ax=ax, label="U/N")
ax = axes[2]
ax.bar(range(1, CFG["n_components"]+1), explained*100, color="steelblue")
ax.set_xlabel("PC"); ax.set_ylabel("Explained Var (%)")
ax.set_title("Scree plot")
ax.axvline(1.5, color="red", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_pca_latent_space.png"), dpi=150)
plt.close()

# Fig 3 — Triple correlation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Triple Correlation: PC1 ↔ log₁₀(Γ) ↔ U/N")
x_fit = np.linspace(pc1.min(), pc1.max(), 200)
ax = axes[0]
sc = ax.scatter(pc1, labels, c=energies, cmap="coolwarm",
                alpha=0.65, s=18, linewidths=0)
fit = np.polyfit(pc1, labels, 1)
ax.plot(x_fit, np.polyval(fit, x_fit), "k-", lw=1.5,
        label=f"R²={r2_pc1_gamma:.4f}")
ax.set_xlabel("PC1"); ax.set_ylabel("log₁₀(Γ)")
ax.set_title("Learned scalar ↔ Cooling rate")
plt.colorbar(sc, ax=ax, label="U/N"); ax.legend()
ax = axes[1]
sc = ax.scatter(pc1, energies, c=labels, cmap="plasma",
                alpha=0.65, s=18, linewidths=0)
fit2 = np.polyfit(pc1, energies, 1)
ax.plot(x_fit, np.polyval(fit2, x_fit), "k-", lw=1.5,
        label=f"R²={r2_pc1_energy:.4f}")
ax.set_xlabel("PC1"); ax.set_ylabel("U/N")
ax.set_title("Learned scalar ↔ Thermodynamic depth")
plt.colorbar(sc, ax=ax, label="log₁₀(Γ)"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_energy_correlation.png"), dpi=150)
plt.close()

# Fig 4 — Violin per rate
fig, ax = plt.subplots(figsize=(12, 6))
data_by_rate = [pc1[labels == r] for r in unique_rates]
vp = ax.violinplot(data_by_rate, positions=range(len(unique_rates)),
                   showmeans=True, showmedians=False)
for body in vp["bodies"]:
    body.set_alpha(0.7)
ax.set_xticks(range(len(unique_rates)))
ax.set_xticklabels([f"{r:.2f}" for r in unique_rates], fontsize=10)
ax.set_xlabel("log₁₀(Γ)"); ax.set_ylabel("PC1")
ax.set_title("PC1 per cooling rate — monotonic = learned fictive temperature")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_violin_per_rate.png"), dpi=150)
plt.close()

# Fig 5 — Adam-Gibbs
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(labels, energies, c=labels, cmap="plasma",
           alpha=0.6, s=20, linewidths=0)
fit3  = np.polyfit(labels, energies, 1)
x3    = np.linspace(labels.min(), labels.max(), 200)
r2_ag = r2_score(energies, np.polyval(fit3, labels))
ax.plot(x3, np.polyval(fit3, x3), "k-", lw=1.5,
        label=f"R²={r2_ag:.4f}")
ax.set_xlabel("log₁₀(Γ)"); ax.set_ylabel("U/N")
ax.set_title("Adam-Gibbs check")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig5_adam_gibbs.png"), dpi=150)
plt.close()
print(f"  Figures saved to {FIG}/")


# ── 13. SAVE ─────────────────────────────────────────────────────────────────
with open(os.path.join(CFG["out_dir"], "regression_cv_results.pkl"), "wb") as f:
    pickle.dump({
        "fold_metrics": fold_metrics,
        "summary": {"r2_mean": float(np.mean(r2_vals)),
                    "r2_std":  float(np.std(r2_vals)),
                    "mae_mean":float(np.mean(mae_vals)),
                    "mae_std": float(np.std(mae_vals))},
        "oof_preds": all_val_preds, "oof_trues": all_val_trues,
        "oof_energies": all_val_energy,
    }, f)

with open(os.path.join(CFG["out_dir"], "pca_results.pkl"), "wb") as f:
    pickle.dump({
        "pca": pca, "scaler": scaler,
        "projections": projections, "latents_raw": all_latents,
        "labels_log10_gamma": labels, "energies": energies,
        "explained_variance": explained,
        "r2_pc1_gamma": float(r2_pc1_gamma),
        "r2_pc1_energy": float(r2_pc1_energy),
    }, f)

print(f"\n  Results saved to {CFG['out_dir']}/")

# ── 14. VERDICT ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  HYPOTHESIS TEST RESULTS")
print("="*60)
h1 = np.mean(r2_vals) > 0.90
h2 = r2_pc1_gamma    > 0.90
h3 = r2_pc1_energy   > 0.90
h4 = explained[0]    > 0.50
print(f"  H1 GNN predicts log10(Γ):     R²={np.mean(r2_vals):.4f}  {'CONFIRMED' if h1 else 'WEAK'}")
print(f"  H2 PC1 ↔ cooling rate:         R²={r2_pc1_gamma:.4f}    {'CONFIRMED' if h2 else 'WEAK'}")
print(f"  H3 PC1 ↔ thermodynamic depth:  R²={r2_pc1_energy:.4f}   {'CONFIRMED' if h3 else 'WEAK'}")
print(f"  H4 PC1 dominant variance:      {explained[0]*100:.1f}%     {'CONFIRMED' if h4 else 'PARTIAL'}")
print("="*60)
print("\nPhase 2 complete.")