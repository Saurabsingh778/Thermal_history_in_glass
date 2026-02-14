import os
# PREVENT JAX FROM HOGGING ALL GPU MEMORY (Keep this, it's good practice)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp
from jax import random, jit, vmap, lax, hessian
from jax_md import space, energy, minimize, simulate, quantity
import numpy as np
import pickle
import time

# --- 1. CONFIGURATION ---
N_PARTICLES = 256  
BOX_SIZE = 15.0    
DIM = 3
BATCH_SIZE = 500   # 1000 Total samples

# --- 2. PHYSICS ENGINE ---
displacement_fn, shift_fn = space.periodic(BOX_SIZE)

def energy_fn(R):
    # Standard LJ potential
    return energy.lennard_jones_pair(displacement_fn, sigma=1.0, epsilon=1.0)(R)

hessian_fn = jit(hessian(energy_fn))

# --- 3. STABILIZED SIMULATION LOOP ---
# We add a Minimizer to fix the "Black Hole" crashes
fire_init, fire_update = minimize.fire_descent(energy_fn, shift_fn)

def run_simulation(key, steps_total):
    key, split = random.split(key)
    R_init = random.uniform(split, (N_PARTICLES, DIM), maxval=BOX_SIZE)
    
    # --- PHASE A: STABILIZATION (The Fix) ---
    # Run 100 steps of energy minimization to remove overlapping particles
    # This prevents the Force -> Infinity explosion
    fire_state = fire_init(R_init)
    fire_state = lax.fori_loop(0, 100, lambda i, s: fire_update(s), fire_state)
    R_stabilized = fire_state.position
    
    # --- PHASE B: PRODUCTION RUN ---
    # Brownian Dynamics 
    dt = 5e-4 # Reduced from 1e-3 to 5e-4 for extra safety
    init_fn, apply_fn = simulate.brownian(energy_fn, shift_fn, dt=dt, kT=2.0)
    state = init_fn(key, R_stabilized) # Start from stabilized coordinates

    def step_fn(i, state_curr):
        progress = i / steps_total
        # Linear cooling: 2.0 -> 0.1
        current_temp = 2.0 - (1.9 * progress)
        return apply_fn(state_curr, kT=current_temp)

    final_state = lax.fori_loop(0, steps_total, step_fn, state)
    
    return final_state.position

# --- 4. FEATURE EXTRACTION ---
def get_local_features(R):
    H = hessian_fn(R)
    H_local = jnp.einsum('iaib->iab', H)
    evals = jnp.linalg.eigvalsh(H_local)
    # Clamp small values to avoid Log(0) errors
    return jnp.log(jnp.maximum(jnp.abs(evals), 1e-6))

# JIT compile
generate_fast = jit(lambda k: run_simulation(k, 2000))    
generate_slow = jit(lambda k: run_simulation(k, 200000)) 

# --- 5. EXECUTION ---
print("Initializing Stabilized Glass Generator...")
key = random.PRNGKey(int(time.time()))

dataset = []
labels = [] 

# FAST BATCH
print(f"Generating {BATCH_SIZE} Fast Glasses (Label 0)...")
for i in range(BATCH_SIZE):
    key, subkey = random.split(key)
    R_final = generate_fast(subkey)
    
    # Check for NaNs immediately during generation
    if jnp.isnan(R_final).any():
        print(f"  WARNING: Nan detected in sample {i}. Retrying...")
        # Simple retry logic (rarely needed with minimization)
        key, subkey = random.split(key)
        R_final = generate_fast(subkey)

    features = get_local_features(R_final)
    
    dataset.append({
        'positions': np.array(R_final), 
        'features': np.array(features)
    })
    labels.append(0)
    if i % 50 == 0: print(f"  Batch {i}/{BATCH_SIZE} done.")

# SLOW BATCH
print(f"Generating {BATCH_SIZE} Slow Glasses (Label 1)...")
for i in range(BATCH_SIZE):
    key, subkey = random.split(key)
    R_final = generate_slow(subkey)
    
    if jnp.isnan(R_final).any():
        print(f"  WARNING: Nan detected in sample {i}. Retrying...")
        key, subkey = random.split(key)
        R_final = generate_slow(subkey)

    features = get_local_features(R_final)
    
    dataset.append({
        'positions': np.array(R_final), 
        'features': np.array(features)
    })
    labels.append(1)
    if i % 50 == 0: print(f"  Batch {i}/{BATCH_SIZE} done.")

with open('glass_challenge_data.pkl', 'wb') as f:
    pickle.dump((dataset, labels), f)

print("\nData Generation Complete. Physics Stabilized.")