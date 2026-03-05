import os
import glob
import jax
# Force float32 to prevent memory waste and silence warnings
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax_md import space
import numpy as np
import pickle
import time
from tqdm.auto import tqdm

# --- 1. CONFIGURATION ---
N_PARTICLES = 4096 
# The GlassBench KA mixture density is rho = 1.2. 
# V = N / rho -> L = V^(1/3)
BOX_SIZE = (4096 / 1.2) ** (1/3) 
DIM = 3

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")
print(f"Dataset Size: N={N_PARTICLES}, Box={BOX_SIZE:.3f}σ")

# --- 2. PHYSICS ENGINE (For Feature Extraction Only) ---
displacement_fn, shift_fn = space.periodic(BOX_SIZE)
disp_vmap = vmap(displacement_fn, in_axes=(None, 0))

# We use the same standard LJ potential for feature extraction
# to keep the geometric probe mathematically consistent with your N=2000 dataset.
def single_particle_energy(r_i, i, R_full):
    dR = disp_vmap(r_i, R_full)
    dist_sq = jnp.sum(dR**2, axis=-1)
    
    # Exclude self
    dist_sq = jnp.where(jnp.arange(N_PARTICLES) == i, 100.0, dist_sq)
    # Soft clip to prevent NaN gradients 
    dist_sq = jnp.where(dist_sq < 0.3, 0.3, dist_sq)
    
    inv_sq = 1.0 / dist_sq
    inv_6 = inv_sq ** 3
    inv_12 = inv_6 ** 2
    return jnp.sum(4.0 * (inv_12 - inv_6))

single_hessian = jax.hessian(single_particle_energy, argnums=0)

@jit
def get_local_features(R):
    def body(i):
        H_i = single_hessian(R[i], i, R)
        evals = jnp.linalg.eigvalsh(H_i)
        return jnp.log(jnp.maximum(jnp.abs(evals), 1e-6))
    return lax.map(body, jnp.arange(N_PARTICLES))

# --- 3. EXECUTION LOOP ---
dataset = []
labels = []

# Grab all .npz files
slow_files = sorted(glob.glob('./KA_Data/T0.44/**/*.npz', recursive=True))
fast_files = sorted(glob.glob('./KA_Data/T0.64/**/*.npz', recursive=True))

# Limit to 500 samples per class to match your original balanced dataset
slow_files = slow_files[:500]
fast_files = fast_files[:500]

print(f"\nFound {len(slow_files)} Slow (T=0.44) and {len(fast_files)} Fast (T=0.64) files.")

# --- Process FAST (T=0.64) ---
print("\nExtracting geometric features for FAST glasses (Label 0)...")
for file_path in tqdm(fast_files, desc="Fast Glasses"):
    # Load native numpy array
    data = np.load(file_path)
    R_coords = data['initial_positions']
    
    # JAX requires jnp array, ensure it fits the periodic box bounds
    R_jnp = jnp.mod(jnp.array(R_coords), BOX_SIZE)
    features = get_local_features(R_jnp)
    
    dataset.append({
        'positions': np.array(R_jnp), 
        'features': np.array(features)
    })
    labels.append(0)

# --- Process SLOW (T=0.44) ---
print("\nExtracting geometric features for SLOW glasses (Label 1)...")
for file_path in tqdm(slow_files, desc="Slow Glasses"):
    data = np.load(file_path)
    R_coords = data['initial_positions']
    
    R_jnp = jnp.mod(jnp.array(R_coords), BOX_SIZE)
    features = get_local_features(R_jnp)
    
    dataset.append({
        'positions': np.array(R_jnp), 
        'features': np.array(features)
    })
    labels.append(1)

# --- Save ---
out_path = 'glass_challenge_data_KA4096.pkl'
with open(out_path, 'wb') as f:
    pickle.dump((dataset, labels), f, protocol=4)

size_mb = os.path.getsize(out_path) / 1e6
print(f"\nData Parsing Complete! Saved {len(dataset)} total samples to {out_path} ({size_mb:.0f} MB).")