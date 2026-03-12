import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── 1. CONFIGURATION FOR KA SYSTEM ───────────────────────────────────────────
KA_DATA_PATH = 'glass_challenge_data_KA4096.pkl'
KA_N_PARTICLES = 4096
KA_BOX_SIZE = (KA_N_PARTICLES / 1.2) ** (1/3)  # ≈ 15.056
RC = 2.5  # Must match the training cutoff exactly

print(f"Loading Kob-Andersen dataset from {KA_DATA_PATH}...")
with open(KA_DATA_PATH, 'rb') as f:
    ka_raw, ka_labels = pickle.load(f)

print(f"Loaded {len(ka_raw)} configurations. Building PyTorch graphs...")

# ── 2. GRAPH CONSTRUCTION (Scale-Invariant) ──────────────────────────────────
# GNNs are size-invariant! We can feed a 4096-node KA graph directly into 
# the model trained on 256-node LJ graphs.
ka_graphs = []
for data, label in zip(ka_raw, ka_labels):
    # Dummy values for u_per_n since we only care about structural inference
    sample_dict = {
        'positions': data['positions'],
        'features': data['features'],
        'log10_gamma': float(label),  # 0.0 for Fast, 1.0 for Slow
        'u_per_n': 0.0 
    }
    
    # Using the build_graph function already defined in your notebook
    g = build_graph(sample_dict, KA_BOX_SIZE, RC)
    ka_graphs.append(g)

# Small batch size to prevent Colab GPU Out-of-Memory with 4096 nodes
ka_loader = make_loader(ka_graphs, batch_size=4, shuffle=False)

# ── 3. ZERO-SHOT INFERENCE ON KA DATA ────────────────────────────────────────
print("\nRunning Zero-Shot Inference on KA mixture...")
model.eval()
ka_latents_list = []
ka_labels_list = []

with torch.no_grad():
    for batch in ka_loader:
        # Extract the latent space using your already-trained LJ model
        _, latent = model(batch, return_latent=True)
        ka_latents_list.append(latent.cpu().numpy())
        ka_labels_list.extend(batch["y"].squeeze().numpy())

ka_latents = np.vstack(ka_latents_list)
ka_labels_arr = np.array(ka_labels_list)

# ── 4. PROJECT INTO THE LJ FICTIVE TEMPERATURE MANIFOLD ──────────────────────
print("Projecting KA graphs into the LJ Structural Fictive Temperature space...")
# Crucial: We use the exact same scaler and PCA fit from the LJ training!
ka_latents_scaled = scaler.transform(ka_latents)
ka_proj = pca.transform(ka_latents_scaled)
ka_pc1 = ka_proj[:, 0]

# Separate by physical state
fast_mask = (ka_labels_arr == 0.0)  # T = 0.64
slow_mask = (ka_labels_arr == 1.0)  # T = 0.44

pc1_fast = ka_pc1[fast_mask]
pc1_slow = ka_pc1[slow_mask]

mean_fast, std_fast = pc1_fast.mean(), pc1_fast.std()
mean_slow, std_slow = pc1_slow.mean(), pc1_slow.std()

# ── 5. STATISTICAL VALIDATION ────────────────────────────────────────────────
# Perform a strictly quantitative statistical test to prove separation
t_stat, p_val = stats.ttest_ind(pc1_slow, pc1_fast, equal_var=False)
cohens_d = (mean_fast - mean_slow) / np.sqrt((std_fast**2 + std_slow**2) / 2)

print("\n" + "="*60)
print("  CROSS-SYSTEM GENERALIZATION RESULTS")
print("="*60)
print(f"  Fast Glass Analog (T=0.64): Mean PC1 = {mean_fast:>8.3f} ± {std_fast:.3f}")
print(f"  Slow Glass Analog (T=0.44): Mean PC1 = {mean_slow:>8.3f} ± {std_slow:.3f}")
print("-" * 60)
print(f"  T-Test p-value      : {p_val:.2e}")
print(f"  Effect Size (Cohen) : {cohens_d:.2f} standard deviations")
print("="*60)

# ── 6. VISUALIZATION ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

parts = ax.violinplot(
    [pc1_fast, pc1_slow],
    positions=[0, 1],
    showmeans=True, showmedians=False
)

colors = ['firebrick', 'navy']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Fast Analog\n(T=0.64)', 'Slow Analog\n(T=0.44)'], fontsize=12)
ax.set_ylabel("LJ-Learned Fictive Temperature (PC1)", fontsize=12)
ax.set_title("Zero-Shot Transfer: Fictive Temperature of KA Binary Mixture\n"
             f"(Effect Size = {cohens_d:.1f}σ, p < 10^{int(np.log10(p_val))})", 
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CFG["out_dir"], "figures", "fig6_cross_system_transfer.png"), dpi=200)
plt.show()

print("\nCross-system validation complete. Figure saved to fig6_cross_system_transfer.png.")