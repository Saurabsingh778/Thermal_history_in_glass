"""
DIAGNOSTIC: What is the ACTUAL signal?
======================================
This will compute simple statistics and show what separates the classes.
"""

import pickle
import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

print("Loading data...")
with open('glass_challenge_data.pkl', 'rb') as f:
    raw_data, raw_labels = pickle.load(f)

# Statistics storage
fast_stats = {'degrees': [], 'avg_dist': [], 'std_dist': [], 'density': []}
slow_stats = {'degrees': [], 'avg_dist': [], 'std_dist': [], 'density': []}

cutoff = 2.5

print("Computing statistics for each glass...")
for i, (item, label) in enumerate(zip(raw_data, raw_labels)):
    pos = item['positions']
    
    # Get all pairwise distances
    dists = pdist(pos)
    
    # Get distances within cutoff (connected neighbors)
    neighbor_dists = dists[dists < cutoff]
    
    if len(neighbor_dists) == 0:
        continue
    
    # Compute statistics
    avg_degree = len(neighbor_dists) / len(pos)  # Average coordination number
    avg_dist = np.mean(neighbor_dists)
    std_dist = np.std(neighbor_dists)
    density = len(neighbor_dists) / (len(pos) * (len(pos) - 1) / 2)  # Fraction of possible edges
    
    # Store by class
    if label == 0:  # Fast
        fast_stats['degrees'].append(avg_degree)
        fast_stats['avg_dist'].append(avg_dist)
        fast_stats['std_dist'].append(std_dist)
        fast_stats['density'].append(density)
    else:  # Slow
        slow_stats['degrees'].append(avg_degree)
        slow_stats['avg_dist'].append(avg_dist)
        slow_stats['std_dist'].append(std_dist)
        slow_stats['density'].append(density)

# Convert to arrays
for key in fast_stats:
    fast_stats[key] = np.array(fast_stats[key])
    slow_stats[key] = np.array(slow_stats[key])

# Print comparison
print("\n" + "="*70)
print("STATISTICAL COMPARISON")
print("="*70)

metrics = [
    ('Average Coordination (Degree)', 'degrees'),
    ('Average Bond Distance', 'avg_dist'),
    ('Std of Bond Distances', 'std_dist'),
    ('Graph Density', 'density')
]

for name, key in metrics:
    fast_mean = np.mean(fast_stats[key])
    slow_mean = np.mean(slow_stats[key])
    diff = abs(fast_mean - slow_mean)
    percent_diff = 100 * diff / fast_mean
    
    print(f"\n{name}:")
    print(f"  Fast Glass: {fast_mean:.4f}")
    print(f"  Slow Glass: {slow_mean:.4f}")
    print(f"  Difference: {diff:.4f} ({percent_diff:.1f}%)")

# Compute separability with simple logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("\n" + "="*70)
print("SIMPLE SEPARABILITY TEST")
print("="*70)

# Test 1: Using ONLY average degree
X_degree = np.concatenate([fast_stats['degrees'], slow_stats['degrees']]).reshape(-1, 1)
y = np.array([0]*len(fast_stats['degrees']) + [1]*len(slow_stats['degrees']))

clf = LogisticRegression()
clf.fit(X_degree, y)
acc_degree = clf.score(X_degree, y)
print(f"\nUsing ONLY average degree: {acc_degree*100:.1f}%")

# Test 2: Using ONLY average distance
X_dist = np.concatenate([fast_stats['avg_dist'], slow_stats['avg_dist']]).reshape(-1, 1)
clf.fit(X_dist, y)
acc_dist = clf.score(X_dist, y)
print(f"Using ONLY average distance: {acc_dist*100:.1f}%")

# Test 3: Using ONLY std of distances
X_std = np.concatenate([fast_stats['std_dist'], slow_stats['std_dist']]).reshape(-1, 1)
clf.fit(X_std, y)
acc_std = clf.score(X_std, y)
print(f"Using ONLY std of distances: {acc_std*100:.1f}%")

# Test 4: Using degree + avg distance + std distance
X_all = np.column_stack([
    np.concatenate([fast_stats['degrees'], slow_stats['degrees']]),
    np.concatenate([fast_stats['avg_dist'], slow_stats['avg_dist']]),
    np.concatenate([fast_stats['std_dist'], slow_stats['std_dist']])
])
clf.fit(X_all, y)
acc_all = clf.score(X_all, y)
print(f"Using ALL geometric features: {acc_all*100:.1f}%")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if acc_degree > 80:
    print("\n⚠️  DEGREE ALONE SEPARATES CLASSES")
    print("    This means one glass type is DENSER.")
    print("    Not a topological difference, just particle packing.")
    
if acc_dist > 80:
    print("\n⚠️  AVERAGE DISTANCE ALONE SEPARATES CLASSES")
    print("    This means one glass has longer/shorter bonds on average.")
    print("    This is a METRIC property, not topology.")
    
if acc_std > 80:
    print("\n⚠️  DISTANCE VARIANCE ALONE SEPARATES CLASSES")
    print("    This means one glass has more uniform bond lengths.")
    print("    This is disorder in the metric, not topology.")

if max(acc_degree, acc_dist, acc_std) > 80:
    print("\n💡 THE SIGNAL IS GEOMETRIC (METRIC), NOT TOPOLOGICAL")
    print("    Your GNN is learning distance patterns, not connectivity patterns.")
    print("    This is still physics, but it's NOT 'pure geometry'.")
else:
    print("\n✅ NO SIMPLE METRIC SEPARATES THE CLASSES")
    print("    The signal requires complex geometric understanding.")
    print("    This validates the GNN approach!")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Degree distribution
axes[0, 0].hist(fast_stats['degrees'], alpha=0.5, bins=30, label='Fast', color='red')
axes[0, 0].hist(slow_stats['degrees'], alpha=0.5, bins=30, label='Slow', color='blue')
axes[0, 0].set_xlabel('Average Degree')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Coordination Number Distribution')
axes[0, 0].legend()

# Plot 2: Average distance
axes[0, 1].hist(fast_stats['avg_dist'], alpha=0.5, bins=30, label='Fast', color='red')
axes[0, 1].hist(slow_stats['avg_dist'], alpha=0.5, bins=30, label='Slow', color='blue')
axes[0, 1].set_xlabel('Average Bond Distance')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Bond Length Distribution')
axes[0, 1].legend()

# Plot 3: Std of distances
axes[1, 0].hist(fast_stats['std_dist'], alpha=0.5, bins=30, label='Fast', color='red')
axes[1, 0].hist(slow_stats['std_dist'], alpha=0.5, bins=30, label='Slow', color='blue')
axes[1, 0].set_xlabel('Std of Bond Distances')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Bond Length Disorder')
axes[1, 0].legend()

# Plot 4: Scatter of degree vs distance
axes[1, 1].scatter(fast_stats['degrees'], fast_stats['avg_dist'], alpha=0.5, label='Fast', color='red')
axes[1, 1].scatter(slow_stats['degrees'], slow_stats['avg_dist'], alpha=0.5, label='Slow', color='blue')
axes[1, 1].set_xlabel('Average Degree')
axes[1, 1].set_ylabel('Average Bond Distance')
axes[1, 1].set_title('Degree vs Distance')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('glass_statistics_diagnostic.png', dpi=150)
print("\n📊 Plots saved to: glass_statistics_diagnostic.png")