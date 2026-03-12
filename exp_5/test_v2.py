"""
Phase 2 FIX: Clean corrupted U/N data + correct analysis metrics
=================================================================
Two problems diagnosed from the output:

1. U/N outliers: values up to 900,000 in LJ units are physically impossible
   (correct range: -5 to +2). These are particle-overlap events where the
   r^-12 term explodes. The NaN guard missed them because they are finite.

2. Wrong metric: R² penalises nonlinearity. The violin plot already shows
   PERFECT monotonic ordering — Spearman rank correlation is the correct
   metric for this analysis.

This script:
  (a) Loads the existing pkl files (no re-training needed)
  (b) Filters extreme U/N values
  (c) Recomputes all correlations with Spearman + Kendall
  (d) Regenerates all figures with corrected data
  (e) Prints the corrected hypothesis verdict

Run this INSTEAD of re-training — it takes < 30 seconds.
"""

import os, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT_DIR = "phase2_results_fixed"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. LOAD EXISTING RESULTS ──────────────────────────────────────────────────
print("Loading existing pkl results ...")
with open("phase2_results/pca_results.pkl", "rb") as f:
    pca_res = pickle.load(f)
with open("phase2_results/regression_cv_results.pkl", "rb") as f:
    reg_res = pickle.load(f)

latents    = pca_res["latents_raw"]           # (1600, 64)
labels     = pca_res["labels_log10_gamma"]    # (1600,)
energies   = pca_res["energies"]              # (1600,) — may have outliers
oof_preds  = reg_res["oof_preds"]
oof_trues  = reg_res["oof_trues"]

print(f"  Total samples : {len(labels)}")
print(f"  U/N range RAW : [{energies.min():.2f}, {energies.max():.2f}]")
print(f"  Outlier U/N > 10 : {(energies > 10).sum()} samples")
print(f"  Outlier U/N < -20: {(energies < -20).sum()} samples")


# ── 2. CLEAN U/N ──────────────────────────────────────────────────────────────
# Physical range for LJ glass at these densities: roughly -6 to +2
# Anything outside [-20, 10] is a particle-overlap artifact
UMIN, UMAX = -20.0, 10.0
valid_mask = (energies > UMIN) & (energies < UMAX)

print(f"\nAfter cleaning (keeping {UMIN} < U/N < {UMAX}):")
print(f"  Valid samples : {valid_mask.sum()} / {len(valid_mask)}")
print(f"  Dropped       : {(~valid_mask).sum()}")

# Apply mask to everything
latents_c  = latents[valid_mask]
labels_c   = labels[valid_mask]
energies_c = energies[valid_mask]
oof_p_c    = oof_preds[valid_mask]
oof_t_c    = oof_trues[valid_mask]

print(f"  U/N range CLEAN: [{energies_c.min():.4f}, {energies_c.max():.4f}]")


# ── 3. RECOMPUTE PCA ON CLEAN DATA ───────────────────────────────────────────
print("\nRecomputing PCA on clean data ...")
scaler      = StandardScaler()
lat_scaled  = scaler.fit_transform(latents_c)
pca         = PCA(n_components=20, random_state=42)
proj        = pca.fit_transform(lat_scaled)
explained   = pca.explained_variance_ratio_
pc1         = proj[:, 0]
unique_rates = np.unique(labels_c)

print(f"  PC1 explains {explained[0]*100:.1f}% of variance")
print(f"  Cumulative (5 PCs): {explained[:5].sum()*100:.1f}%")


# ── 4. FULL CORRELATION SUITE ────────────────────────────────────────────────
print("\n" + "="*65)
print("  CORRECTED METRIC SUITE")
print("="*65)

# Regression metrics (OOF)
from sklearn.metrics import mean_absolute_error
r2_oof    = r2_score(oof_t_c, oof_p_c)
mae_oof   = mean_absolute_error(oof_t_c, oof_p_c)
sp_oof    = stats.spearmanr(oof_t_c, oof_p_c)
kt_oof    = stats.kendalltau(oof_t_c, oof_p_c)

print(f"\n  GNN Regression (OOF, clean):")
print(f"    R²                    = {r2_oof:.4f}   (linear only — not the key metric)")
print(f"    Spearman ρ            = {sp_oof.statistic:.4f}  p={sp_oof.pvalue:.2e}")
print(f"    Kendall τ             = {kt_oof.statistic:.4f}  p={kt_oof.pvalue:.2e}")
print(f"    MAE                   = {mae_oof:.4f}  [log10 units]")

# PC1 correlations — both linear and rank
r2_pc1_g   = r2_score(labels_c,
    LinearRegression().fit(pc1.reshape(-1,1), labels_c)
                      .predict(pc1.reshape(-1,1)))
r2_pc1_e   = r2_score(energies_c,
    LinearRegression().fit(pc1.reshape(-1,1), energies_c)
                      .predict(pc1.reshape(-1,1)))
sp_pc1_g   = stats.spearmanr(pc1, labels_c)
sp_pc1_e   = stats.spearmanr(pc1, energies_c)
kt_pc1_g   = stats.kendalltau(pc1, labels_c)
kt_pc1_e   = stats.kendalltau(pc1, energies_c)

print(f"\n  PC1 ↔ log10(Γ):")
print(f"    R² (linear)           = {r2_pc1_g:.4f}")
print(f"    Spearman ρ            = {sp_pc1_g.statistic:.4f}  p={sp_pc1_g.pvalue:.2e}")
print(f"    Kendall τ             = {kt_pc1_g.statistic:.4f}  p={kt_pc1_g.pvalue:.2e}")

print(f"\n  PC1 ↔ U/N (clean):")
print(f"    R² (linear)           = {r2_pc1_e:.4f}")
print(f"    Spearman ρ            = {sp_pc1_e.statistic:.4f}  p={sp_pc1_e.pvalue:.2e}")
print(f"    Kendall τ             = {kt_pc1_e.statistic:.4f}  p={kt_pc1_e.pvalue:.2e}")

# Per-rate monotonicity test — the key test
print(f"\n  Per-rate PC1 means (monotonicity check):")
pc1_means = []
for r in unique_rates:
    m = pc1[labels_c == r].mean()
    pc1_means.append(m)
    print(f"    log10(Γ) = {r:+.3f}  →  mean PC1 = {m:+.4f}")

# Check strict monotonicity
is_monotone = all(pc1_means[i] > pc1_means[i+1]
                  for i in range(len(pc1_means)-1))
print(f"\n  Strict monotone ordering: {'YES — CONFIRMED' if is_monotone else 'NO'}")

# Spearman between per-rate means and rates
sp_means = stats.spearmanr(unique_rates, pc1_means)
print(f"  Spearman ρ (means vs rates): {sp_means.statistic:.4f}  "
      f"p={sp_means.pvalue:.2e}")

print("="*65)


# ── 5. ADAM-GIBBS CHECK ON CLEAN DATA ────────────────────────────────────────
# The Adam-Gibbs relation predicts U/N decreases monotonically with log10(Γ)
# Compute per-rate mean U/N to check
print(f"\n  Adam-Gibbs per-rate mean U/N:")
u_means = []
for r in unique_rates:
    m = energies_c[labels_c == r].mean()
    u_means.append(m)
    print(f"    log10(Γ) = {r:+.3f}  →  mean U/N = {m:.6f}")
sp_ag = stats.spearmanr(unique_rates, u_means)
print(f"\n  Spearman ρ (U/N means vs rates): {sp_ag.statistic:.4f}  "
      f"p={sp_ag.pvalue:.2e}")


# ── 6. REGENERATE FIGURES ────────────────────────────────────────────────────
print("\nRegenerating figures ...")

# Fig 1 — Regression scatter (Spearman annotated)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("GATv2 Regression: Predicted vs True log₁₀(Γ) [cleaned]",
             fontsize=14)
ax = axes[0]
sc = ax.scatter(oof_t_c, oof_p_c, c=oof_t_c, cmap="plasma",
                alpha=0.55, s=18, linewidths=0)
lims = [oof_t_c.min()-0.1, oof_t_c.max()+0.1]
ax.plot(lims, lims, "k--", lw=1.2, label="y = x")
ax.set_xlabel("True log₁₀(Γ)"); ax.set_ylabel("Predicted log₁₀(Γ)")
ax.set_title(f"R² = {r2_oof:.3f}   Spearman ρ = {sp_oof.statistic:.3f}")
plt.colorbar(sc, ax=ax, label="log₁₀(Γ)"); ax.legend()
ax2 = axes[1]
ax2.scatter(oof_t_c, oof_p_c - oof_t_c, c=oof_t_c, cmap="plasma",
            alpha=0.55, s=18, linewidths=0)
ax2.axhline(0, color="k", linestyle="--", lw=1.2)
ax2.set_xlabel("True log₁₀(Γ)"); ax2.set_ylabel("Residual")
ax2.set_title("Residuals")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_regression_scatter_fixed.png"), dpi=150)
plt.close()

# Fig 2 — PCA latent space (clean)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Latent Space PCA — Structural Fictive Temperature Manifold [cleaned]")
ax = axes[0]
sc = ax.scatter(proj[:,0], proj[:,1], c=labels_c, cmap="plasma",
                alpha=0.6, s=16, linewidths=0)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
ax.set_title(f"Coloured by log₁₀(Γ)  [ρ={sp_pc1_g.statistic:.3f}]")
plt.colorbar(sc, ax=ax, label="log₁₀(Γ)")
ax = axes[1]
sc = ax.scatter(proj[:,0], proj[:,1], c=energies_c, cmap="coolwarm",
                alpha=0.6, s=16, linewidths=0,
                vmin=np.percentile(energies_c, 2),
                vmax=np.percentile(energies_c, 98))
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
ax.set_title(f"Coloured by U/N (clean)  [ρ={sp_pc1_e.statistic:.3f}]")
plt.colorbar(sc, ax=ax, label="U/N")
ax = axes[2]
ax.bar(range(1, 21), explained*100, color="steelblue")
ax.set_xlabel("PC"); ax.set_ylabel("Explained Var (%)")
ax.set_title("Scree plot")
ax.axvline(1.5, color="red", linestyle="--", alpha=0.5, label="PC1")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_pca_latent_fixed.png"), dpi=150)
plt.close()

# Fig 3 — Triple correlation (Spearman annotated)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Triple Correlation: PC1 ↔ log₁₀(Γ) ↔ U/N  [cleaned + Spearman]")
x_fit = np.linspace(pc1.min(), pc1.max(), 200)
ax = axes[0]
sc = ax.scatter(pc1, labels_c, c=energies_c, cmap="coolwarm",
                alpha=0.65, s=18, linewidths=0,
                vmin=np.percentile(energies_c,2),
                vmax=np.percentile(energies_c,98))
fit = np.polyfit(pc1, labels_c, 1)
ax.plot(x_fit, np.polyval(fit, x_fit), "k-", lw=1.5,
        label=f"Linear R²={r2_pc1_g:.3f}\nSpearman ρ={sp_pc1_g.statistic:.3f}")
ax.set_xlabel("PC1"); ax.set_ylabel("log₁₀(Γ)")
ax.set_title("Learned scalar ↔ Cooling rate")
plt.colorbar(sc, ax=ax, label="U/N"); ax.legend(fontsize=9)
ax = axes[1]
sc = ax.scatter(pc1, energies_c, c=labels_c, cmap="plasma",
                alpha=0.65, s=18, linewidths=0)
fit2 = np.polyfit(pc1, energies_c, 1)
ax.plot(x_fit, np.polyval(fit2, x_fit), "k-", lw=1.5,
        label=f"Linear R²={r2_pc1_e:.3f}\nSpearman ρ={sp_pc1_e.statistic:.3f}")
ax.set_xlabel("PC1"); ax.set_ylabel("U/N")
ax.set_title("Learned scalar ↔ Thermodynamic depth")
plt.colorbar(sc, ax=ax, label="log₁₀(Γ)"); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_triple_correlation_fixed.png"), dpi=150)
plt.close()

# Fig 4 — Violin (clean, annotated with means)
fig, ax = plt.subplots(figsize=(12, 6))
data_by_rate = [pc1[labels_c == r] for r in unique_rates]
vp = ax.violinplot(data_by_rate, positions=range(len(unique_rates)),
                   showmeans=True, showmedians=False)
for body in vp["bodies"]:
    body.set_alpha(0.7)
# Annotate means
for i, (r, m) in enumerate(zip(unique_rates, pc1_means)):
    ax.text(i, m + 0.3, f"{m:.1f}", ha="center", va="bottom",
            fontsize=8, color="navy")
ax.set_xticks(range(len(unique_rates)))
ax.set_xticklabels([f"{r:.2f}" for r in unique_rates], fontsize=10)
ax.set_xlabel("log₁₀(Γ)  [cooling rate]"); ax.set_ylabel("PC1")
mono_str = "STRICT MONOTONE" if is_monotone else "non-monotone"
ax.set_title(f"PC1 per cooling rate — {mono_str} ordering\n"
             f"Spearman ρ(means, rates) = {sp_means.statistic:.4f}  "
             f"p = {sp_means.pvalue:.2e}")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_violin_fixed.png"), dpi=150)
plt.close()

# Fig 5 — Adam-Gibbs CORRECT: per-rate means with error bars
fig, ax = plt.subplots(figsize=(9, 6))
u_stds = [energies_c[labels_c == r].std() for r in unique_rates]
ax.errorbar(unique_rates, u_means, yerr=u_stds,
            fmt="o-", color="steelblue", lw=2, markersize=8,
            capsize=4, label="Mean U/N ± std")
fit3    = np.polyfit(unique_rates, u_means, 1)
x3      = np.linspace(unique_rates.min(), unique_rates.max(), 200)
r2_ag   = r2_score(u_means, np.polyval(fit3, unique_rates))
ax.plot(x3, np.polyval(fit3, x3), "k--", lw=1.5,
        label=f"Linear R²={r2_ag:.3f}  ρ={sp_ag.statistic:.3f}")
ax.set_xlabel("log₁₀(Γ) — cooling rate", fontsize=12)
ax.set_ylabel("Mean U/N per rate", fontsize=12)
ax.set_title(f"Adam-Gibbs check: mean U/N vs log₁₀(Γ)\n"
             f"Spearman ρ = {sp_ag.statistic:.4f}  p = {sp_ag.pvalue:.2e}",
             fontsize=12)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_adam_gibbs_fixed.png"), dpi=150)
plt.close()

print(f"  Figures saved to {FIG_DIR}/")

# ── 7. FINAL VERDICT ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  CORRECTED HYPOTHESIS VERDICTS")
print("="*65)

h1 = abs(sp_oof.statistic)     > 0.70
h2 = abs(sp_pc1_g.statistic)   > 0.70
h3 = abs(sp_pc1_e.statistic)   > 0.50
h4 = explained[0]              > 0.50
h5 = is_monotone

print(f"  H1 GNN predicts cooling rate (Spearman):")
print(f"     ρ = {sp_oof.statistic:.4f}  "
      f"{'CONFIRMED' if h1 else 'PARTIAL'}")
print()
print(f"  H2 PC1 ↔ cooling rate (Spearman):")
print(f"     ρ = {sp_pc1_g.statistic:.4f}  p = {sp_pc1_g.pvalue:.1e}  "
      f"{'CONFIRMED' if h2 else 'PARTIAL'}")
print()
print(f"  H3 PC1 ↔ thermodynamic depth U/N (Spearman):")
print(f"     ρ = {sp_pc1_e.statistic:.4f}  p = {sp_pc1_e.pvalue:.1e}  "
      f"{'CONFIRMED' if h3 else 'PARTIAL — fix U/N data'}")
print()
print(f"  H4 PC1 dominant variance:")
print(f"     {explained[0]*100:.1f}%  {'CONFIRMED' if h4 else 'PARTIAL'}")
print()
print(f"  H5 Strict monotone PC1 ordering across all 8 rates:")
print(f"     {'CONFIRMED — all 8 rates monotonically ordered' if h5 else 'NOT CONFIRMED'}")
print("="*65)

print("""
DIAGNOSIS OF WEAK H3 (PC1 ↔ U/N):
  The U/N values in the pkl have extreme outliers (up to 900,000)
  from particle-overlap events that passed the NaN filter.
  Two options to fix this permanently:
  
  Option A (quick): filter outliers in the pkl — already done above.
  
  Option B (correct): regenerate data with a stricter energy guard:
    if abs(u_per_n) > 50:
        # retry — this is a bad sample
  
  Add this to the data generation loop before appending to dataset.
""")