import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
from jax import random, jit, lax
from jax_md import space, energy, simulate, quantity
import numpy as np
import pickle
import time

print("="*70)
print("GENERATING ALIEN DATA: KOB-ANDERSEN METALLIC GLASS")
print("Target: 80:20 Binary Mixture (Ni80P20 analog)")
print("="*70)

# --- 1. CONFIGURATION ---
N = 500  # Total particles
N_A = 400 # 80% Large (Type 0)
N_B = 100 # 20% Small (Type 1)
BOX_SIZE = 15.0 # High density (~1.2)
DIM = 3
BATCH_SIZE = 20 # Generate 20 samples (10 Fast, 10 Slow) for testing

# --- 2. KOB-ANDERSEN PHYSICS ---
# These parameters define a Metallic Glass
# sigma_AA = 1.0, sigma_AB = 0.8, sigma_BB = 0.88
# epsilon_AA = 1.0, epsilon_AB = 1.5, epsilon_BB = 0.5
sigmas = jnp.array([[1.0, 0.8], [0.8, 0.88]])
epsilons = jnp.array([[1.0, 1.5], [1.5, 0.5]])

species = jnp.where(jnp.arange(N) < N_A, 0, 1) # 0 = A, 1 = B

displacement_fn, shift_fn = space.periodic(BOX_SIZE)

# Neighbor list needed for binary potential efficiency? 
# For N=500, dense pair matrix is fine and simpler.
def energy_fn(R):
    return energy.lennard_jones_pair(
        displacement_fn, 
        species=species, 
        sigma=sigmas, 
        epsilon=epsilons
    )(R)

# --- 3. SIMULATION LOOP ---
def run_simulation(key, steps_total):
    key, split = random.split(key)
    R_init = random.uniform(split, (N, DIM), maxval=BOX_SIZE)
    
    # 2.0 -> 0.1 Cooling
    dt = 5e-4
    init_fn, apply_fn = simulate.brownian(energy_fn, shift_fn, dt=dt, kT=2.0)
    state = init_fn(key, R_init)

    def step_fn(i, state_curr):
        progress = i / steps_total
        # Linear cooling
        current_temp = 2.0 - (1.9 * progress)
        return apply_fn(state_curr, kT=current_temp)

    final_state = lax.fori_loop(0, steps_total, step_fn, state)
    return final_state.position

generate_fast = jit(lambda k: run_simulation(k, 5000))    # Fast Quench
generate_slow = jit(lambda k: run_simulation(k, 500000))  # Slow Anneal (100x)

# --- 4. EXECUTION ---
key = random.PRNGKey(int(time.time()))
dataset = []
labels = []

print("Simulating Metallic Glasses...")

# Generate FAST
for i in range(10):
    print(f"  Generating Fast Alloy {i+1}/10...")
    key, subkey = random.split(key)
    R = generate_fast(subkey)
    dataset.append({'positions': np.array(R)})
    labels.append(0)

# Generate SLOW
for i in range(10):
    print(f"  Generating Slow Alloy {i+1}/10...")
    key, subkey = random.split(key)
    R = generate_slow(subkey)
    dataset.append({'positions': np.array(R)})
    labels.append(1)

with open('ka_glass_test_data.pkl', 'wb') as f:
    pickle.dump((dataset, labels), f)

print("\nAlien Data Generated.")