import os
# PREVENT JAX FROM HOGGING ALL GPU MEMORY
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
# Float32 is strongly recommended here for speed
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
from jax import random, jit, lax, hessian
from jax_md import space, energy, minimize, simulate
import numpy as np
import pickle
import time
from tqdm.auto import tqdm

# --- 1. CONFIGURATION ---
N_PARTICLES = 256  
BOX_SIZE = 15.0    
DIM = 3
SAMPLES_PER_RATE = 200  

# Thermodynamics
T_START = 2.0
T_END = 0.1
DELTA_T = T_START - T_END
DT = 5e-4

# Logarithmically spaced cooling rates (Γ)
COOLING_RATES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")

# --- 2. PHYSICS ENGINE ---
displacement_fn, shift_fn = space.periodic(BOX_SIZE)

def energy_fn(R):
    return energy.lennard_jones_pair(displacement_fn, sigma=1.0, epsilon=1.0)(R)

hessian_fn = jit(hessian(energy_fn))

# Soft potential for stabilization
def soft_energy_fn(R):
    return energy.soft_sphere_pair(displacement_fn, sigma=1.0)(R)

fire_init, fire_update = minimize.fire_descent(soft_energy_fn, shift_fn)

# --- 3. JIT FACTORY FOR SIMULATION ---
def make_generator_for_rate(gamma):
    steps_total = int(DELTA_T / (gamma * DT))
    
    @jit
    def generate_single_glass(key):
        key, split = random.split(key)
        R_init = random.uniform(split, (N_PARTICLES, DIM), maxval=BOX_SIZE)
        
        # --- PHASE A: STABILIZATION ---
        fire_state = fire_init(R_init)
        fire_state = lax.fori_loop(0, 100, lambda i, s: fire_update(s), fire_state)
        R_stabilized = fire_state.position
        
        # --- PHASE B: PRODUCTION RUN ---
        init_fn, apply_fn = simulate.brownian(energy_fn, shift_fn, dt=DT, kT=T_START)
        state = init_fn(key, R_stabilized)

        def step_fn(i, state_curr):
            progress = i / steps_total
            current_temp = T_START - (DELTA_T * progress)
            return apply_fn(state_curr, kT=current_temp)

        final_state = lax.fori_loop(0, steps_total, step_fn, state)
        R_final = final_state.position
        
        # --- PHASE C: THERMODYNAMICS & GEOMETRY ---
        total_energy = energy_fn(R_final)
        u_per_n = total_energy / N_PARTICLES
        
        H = hessian_fn(R_final)
        H_local = jnp.einsum('iaib->iab', H)
        evals = jnp.linalg.eigvalsh(H_local)
        features = jnp.log(jnp.maximum(jnp.abs(evals), 1e-6))
        
        return R_final, features, u_per_n

    return generate_single_glass, steps_total

# --- 4. EXECUTION LOOP ---
print("\nInitializing Fast Single-Sample Generators...")
generators = {}
for gamma in COOLING_RATES:
    gen_fn, steps = make_generator_for_rate(gamma)
    generators[gamma] = {'fn': gen_fn, 'steps': steps}
    print(f"  Γ = {gamma:<7} -> {steps:>8} simulation steps")

key = random.PRNGKey(int(time.time()))
dataset = []

print("\nCommencing Dataset Generation...")
for gamma in COOLING_RATES:
    gen_fn = generators[gamma]['fn']
    
    # Use tqdm to show a live progress bar for each cooling rate
    print(f"\nProcessing Cooling Rate: Γ = {gamma}")
    for i in tqdm(range(SAMPLES_PER_RATE), desc=f"Γ={gamma}"):
        key, subkey = random.split(key)
        R_final, features, u_per_n = gen_fn(subkey)
        
        # NaN Guard
        if jnp.isnan(R_final).any():
            print(f"\n  WARNING: NaN detected in sample {i}. Retrying...")
            key, subkey = random.split(key)
            R_final, features, u_per_n = gen_fn(subkey)
            
        dataset.append({
            'positions': np.array(R_final),
            'features': np.array(features),
            'u_per_n': float(u_per_n),
            'log10_gamma': float(np.log10(gamma))
        })

# --- 5. SAVE TO DISK ---
out_path = 'fictive_temp_dataset.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(dataset, f, protocol=4)

size_mb = os.path.getsize(out_path) / 1e6
print(f"\nData Generation Complete! Saved {len(dataset)} samples to {out_path} ({size_mb:.1f} MB).")