import os
# PREVENT JAX FROM HOGGING ALL GPU MEMORY
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
# Enable 64-bit precision for better stability during minimization (optional on TPU but recommended)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, jit, lax
from jax_md import space, energy, minimize, simulate
import numpy as np
import pickle
import time

# --- 1. CONFIGURATION ---
N_PARTICLES = 256    
BOX_SIZE = 15.0      
DIM = 3
N_SAMPLES = 1000     
SCRAMBLE_STEPS = 2000 
QUENCH_STEPS = 2000  
SCRAMBLE_TEMP = 0.8

# --- 2. SHAPE GENERATOR ---
def get_s_curve_positions(key, n_particles, box_size):
    """Generates particles along a thickened 3D 'S' tube."""
    t = jnp.linspace(-np.pi, np.pi, n_particles)
    
    scale = box_size * 0.25
    center = box_size / 2.0
    
    # Base Curve
    x = jnp.sin(t) * scale + center
    y = (t / np.pi) * scale + center
    # Initial Z spread
    z = jnp.ones_like(t) * center
    
    R = jnp.stack([x, y, z], axis=1)
    
    # Add thicker noise to prevent initial overlaps
    # Increased noise scale from 0.2 to 0.5 to give atoms breathing room
    key, subkey = random.split(key)
    noise = random.normal(subkey, (n_particles, 3)) * 0.5 
    R = R + noise
    
    return jnp.mod(R, box_size)

# --- 3. DUAL PHYSICS ENGINES ---
displacement_fn, shift_fn = space.periodic(BOX_SIZE)

# A. SOFT POTENTIAL (For De-overlapping)
# This potential is finite at r=0. It gently pushes overlapping atoms apart.
def soft_energy_fn(R):
    return energy.soft_sphere_pair(displacement_fn, sigma=1.0)(R)

# B. HARD POTENTIAL (Real Physics)
# The standard LJ potential. Singular at r=0.
def lj_energy_fn(R):
    return energy.lennard_jones_pair(displacement_fn, sigma=1.0, epsilon=1.0)(R)

# --- 4. ROBUST SIMULATION LOOP ---
# We define TWO minimizers
fire_init_soft, fire_update_soft = minimize.fire_descent(soft_energy_fn, shift_fn)
fire_init_hard, fire_update_hard = minimize.fire_descent(lj_energy_fn, shift_fn)

def run_scramble_and_quench(key, R_start):
    
    # --- PHASE 0: SOFT WARMUP (The Fix) ---
    # Use Soft Sphere to resolve overlaps without NaN explosion
    fire_state = fire_init_soft(R_start)
    # Run 100 steps to gently push atoms to r > 0.8
    fire_state = lax.fori_loop(0, 100, lambda i, s: fire_update_soft(s), fire_state)
    R_soft = fire_state.position
    
    # --- PHASE A: HARD MINIMIZATION ---
    # Now that atoms are separated, switch to real LJ physics
    fire_state = fire_init_hard(R_soft)
    fire_state = lax.fori_loop(0, 100, lambda i, s: fire_update_hard(s), fire_state)
    R_stable = fire_state.position
    
    # --- PHASE B: ENTROPY INJECTION (Melting) ---
    # High Temp (3.0). Reduced dt slightly for safety.
    dt = 1e-3 
    init_fn, apply_fn = simulate.brownian(lj_energy_fn, shift_fn, dt=dt, kT=SCRAMBLE_TEMP) # Use 0.8
    state = init_fn(key, R_stable)
    
    # Run Scramble
    state = lax.fori_loop(0, SCRAMBLE_STEPS, lambda i, s: apply_fn(s), state)
    
    # --- PHASE C: MEMORY ENCODING (Quenching) ---
    def cool_step(i, state_curr):
        T = 3.0 - (2.9 * (i / QUENCH_STEPS))
        return apply_fn(state_curr, kT=T)
        
    final_state = lax.fori_loop(0, QUENCH_STEPS, cool_step, state)
    
    # Return the clean (stabilized) target and the messy glass
    return R_stable, final_state.position 

# JIT Compile
run_fast = jit(run_scramble_and_quench)

# --- 5. HESSIAN FEATURES ---
hessian_fn = jit(jax.hessian(lj_energy_fn))

def get_hessian_features(R):
    H = hessian_fn(R)
    # Extract 3x3 diagonal blocks
    H_local = jnp.einsum('iaib->iab', H) 
    evals = jnp.linalg.eigvalsh(H_local)
    # Clamp to avoid Log(0)
    return jnp.log(jnp.maximum(jnp.abs(evals), 1e-6))

get_features_fast = jit(get_hessian_features)

# --- 6. EXECUTION ---
print(f"Initializing Robust Time-Reversal Experiment...")
key = random.PRNGKey(int(time.time()))
dataset = []

start_time = time.time()

for i in range(N_SAMPLES):
    key, shape_key, sim_key = random.split(key, 3)
    
    # 1. Generate Target
    R_target_raw = get_s_curve_positions(shape_key, N_PARTICLES, BOX_SIZE)
    
    # 2. Run Simulation
    R_target_clean, R_input_glass = run_fast(sim_key, R_target_raw)
    
    # 3. Validation
    if jnp.isnan(R_input_glass).any():
        print(f"WARNING: NaN in sample {i} even after fix. Retrying...")
        continue
        
    features = get_features_fast(R_input_glass)
    
    dataset.append({
        'pos_glass': np.array(R_input_glass),   
        'features': np.array(features),         
        'pos_hidden': np.array(R_target_clean)  
    })
    
    if i % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Sample {i}/{N_SAMPLES} | Time: {elapsed:.1f}s")

# Save
with open('time_reversal_data.pkl', 'wb') as f:
    pickle.dump(dataset, f)

print(f"\nSuccess.")