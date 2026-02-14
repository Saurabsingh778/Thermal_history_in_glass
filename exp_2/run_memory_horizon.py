import os
# PREVENT JAX FROM HOGGING ALL GPU MEMORY
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, jit, lax
from jax_md import space, energy, minimize, simulate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv 
import matplotlib.pyplot as plt
import time
import pandas as pd

# --- CONFIGURATION ---
# The "X-Axis" of your plot: Increasing Entropy
STEPS_LIST = [0, 500, 1000, 2000, 3000, 4000, 5000] 

# Experiment Settings (Reduced for speed, but valid for trend)
N_SAMPLES_PER_POINT = 200 
EPOCHS_PER_POINT = 50     
BATCH_SIZE = 16
LR = 0.001
BOX_SIZE = 15.0
RADIUS = 2.5
N_PARTICLES = 256
SCRAMBLE_TEMP = 0.8
QUENCH_STEPS = 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. JAX PHYSICS ENGINE (Robust) ---
displacement_fn, shift_fn = space.periodic(BOX_SIZE)

def soft_energy_fn(R):
    return energy.soft_sphere_pair(displacement_fn, sigma=1.0)(R)

def lj_energy_fn(R):
    return energy.lennard_jones_pair(displacement_fn, sigma=1.0, epsilon=1.0)(R)

fire_init_soft, fire_update_soft = minimize.fire_descent(soft_energy_fn, shift_fn)
fire_init_hard, fire_update_hard = minimize.fire_descent(lj_energy_fn, shift_fn)

hessian_fn = jit(jax.hessian(lj_energy_fn))

def get_hessian_features(R):
    H = hessian_fn(R)
    H_local = jnp.einsum('iaib->iab', H) 
    evals = jnp.linalg.eigvalsh(H_local)
    return jnp.log(jnp.maximum(jnp.abs(evals), 1e-6))

def run_simulation_logic(key, R_start, n_scramble):
    # 1. Soft Warmup
    fire_state = fire_init_soft(R_start)
    fire_state = lax.fori_loop(0, 100, lambda i, s: fire_update_soft(s), fire_state)
    R_soft = fire_state.position
    
    # 2. Hard Minimization
    fire_state = fire_init_hard(R_soft)
    fire_state = lax.fori_loop(0, 100, lambda i, s: fire_update_hard(s), fire_state)
    R_stable = fire_state.position
    
    # 3. Scramble (Entropy Injection)
    # If steps=0, we skip this and just retain the 'clean' shape (Perfect Memory)
    if n_scramble > 0:
        dt = 1e-3 
        init_fn, apply_fn = simulate.brownian(lj_energy_fn, shift_fn, dt=dt, kT=SCRAMBLE_TEMP)
        state = init_fn(key, R_stable)
        state = lax.fori_loop(0, n_scramble, lambda i, s: apply_fn(s), state)
        
        # 4. Quench (Memory Encoding)
        def cool_step(i, state_curr):
            T = 3.0 - (2.9 * (i / QUENCH_STEPS)) 
            return apply_fn(state_curr, kT=T)

        final_state = lax.fori_loop(0, QUENCH_STEPS, cool_step, state)
        return R_stable, final_state.position
    else:
        # 0 Steps: The glass is the target (trivial identity mapping)
        return R_stable, R_stable

# JIT the simulation with 'n_scramble' as static to allow loop unrolling
run_fast = jit(run_simulation_logic, static_argnums=(2,))
get_features_fast = jit(get_hessian_features)

def get_s_curve_positions(key, n_particles, box_size):
    t = jnp.linspace(-np.pi, np.pi, n_particles)
    scale = box_size * 0.25
    center = box_size / 2.0
    x = jnp.sin(t) * scale + center
    y = (t / np.pi) * scale + center
    z = jnp.ones_like(t) * center
    R = jnp.stack([x, y, z], axis=1)
    key, subkey = random.split(key)
    noise = random.normal(subkey, (n_particles, 3)) * 0.5 
    return jnp.mod(R + noise, box_size)

# --- 2. PYTORCH UTILS ---
def simple_radius_graph(pos, r, loop=False):
    dist = torch.cdist(pos, pos) 
    mask = dist < r
    if not loop: mask.fill_diagonal_(False)
    return mask.nonzero(as_tuple=False).t()

class GlassDatasetList(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

class GlassTimeReversal(nn.Module):
    def __init__(self):
        super().__init__()
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

def centered_mse_loss(pred_pos, target_pos, batch_indices):
    from torch_geometric.utils import scatter
    pred_com = scatter(pred_pos, batch_indices, dim=0, reduce='mean')
    target_com = scatter(target_pos, batch_indices, dim=0, reduce='mean')
    pred_centered = pred_pos - pred_com[batch_indices]
    target_centered = target_pos - target_com[batch_indices]
    return F.mse_loss(pred_centered, target_centered)

# --- 3. MAIN LOOP ---
results_mse = []
key = random.PRNGKey(int(time.time()))

print("Starting Memory Horizon Experiment...")
print(f"Testing Scramble Steps: {STEPS_LIST}")

for steps in STEPS_LIST:
    print(f"\n--- Phase A: Generating Data for {steps} Steps ---")
    data_list = []
    generated = 0
    start_t = time.time()
    
    while generated < N_SAMPLES_PER_POINT:
        key, shape_key, sim_key = random.split(key, 3)
        R_target_raw = get_s_curve_positions(shape_key, N_PARTICLES, BOX_SIZE)
        
        # Run Sim
        R_target_clean, R_input_glass = run_fast(sim_key, R_target_raw, steps)
        
        if jnp.isnan(R_input_glass).any(): continue
        
        # Features
        features = get_features_fast(R_input_glass)
        
        # To Torch
        pos_glass = torch.tensor(np.array(R_input_glass), dtype=torch.float) / BOX_SIZE
        pos_target = torch.tensor(np.array(R_target_clean), dtype=torch.float) / BOX_SIZE
        hessian_feats = torch.tensor(np.array(features), dtype=torch.float)
        
        edge_index = simple_radius_graph(pos_glass, r=RADIUS/BOX_SIZE, loop=False)
        row, col = edge_index
        edge_attr = (pos_glass[row] - pos_glass[col]).norm(dim=-1).unsqueeze(-1)
        
        data = Data(x=hessian_feats, pos=pos_glass, edge_index=edge_index, edge_attr=edge_attr, y=pos_target)
        data_list.append(data)
        generated += 1
        
    print(f"  Generated {generated} samples in {time.time()-start_t:.1f}s")
    
    print(f"--- Phase B: Training Model for {steps} Steps ---")
    dataset = GlassDatasetList(data_list)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Re-initialize model for fair comparison
    model = GlassTimeReversal().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS_PER_POINT):
        model.train()
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
        
        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            
    print(f"  Best Loss (MSE): {best_loss:.6f}")
    results_mse.append(best_loss)

# --- 4. PLOTTING ---
print("\nPlotting Memory Horizon...")
plt.figure(figsize=(10, 6))

# Main Curve
plt.plot(STEPS_LIST, results_mse, 'o-', linewidth=3, color='#6a0dad', label='Reconstruction Error')

# Baselines
plt.axhline(y=0.083, color='gray', linestyle='--', label='Random Baseline (Liquid Limit)')
plt.axhline(y=0.0, color='green', linestyle=':', label='Perfect Memory (Solid)')

# Annotations
plt.text(STEPS_LIST[1], results_mse[1] + 0.005, 'Glassy Memory', color='purple')
if len(results_mse) > 4:
    plt.text(STEPS_LIST[-1], results_mse[-1] - 0.005, 'Liquid Chaos', color='gray', ha='right')

plt.title('The Memory Horizon: Half-Life of Geometric Information', fontsize=14)
plt.xlabel('Entropy Injection (Scramble Steps at T=0.8)', fontsize=12)
plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig('memory_horizon.png')
print("Saved 'memory_horizon.png'.")

# Save Data
df = pd.DataFrame({'Steps': STEPS_LIST, 'MSE': results_mse})
df.to_csv('memory_horizon_results.csv', index=False)
print(df)