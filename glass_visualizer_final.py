import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as colors

print("="*70)
print("VISUALIZING THE GEOMETRIC MEMORY (HIGH RES)")
print("="*70)

# Use Dark Background for "Neon" effect (High Contrast)
plt.style.use('dark_background')

# Load Data
with open('glass_challenge_data.pkl', 'rb') as f:
    raw_data, raw_labels = pickle.load(f)

# Get one Fast and one Slow sample
fast_idx = raw_labels.index(0)
slow_idx = raw_labels.index(1)

def plot_strain_network(ax, item, title):
    pos = item['positions']
    cutoff = 2.5
    
    # 1. NORMALIZE VOLUME
    dists = pdist(pos)
    valid_dists = dists[dists < cutoff]
    avg_bond = np.mean(valid_dists)
    
    scale = 1.0 / avg_bond
    pos = pos * scale
    scaled_cutoff = cutoff * scale
    
    # 2. CALCULATE STRAIN
    dist_mat = squareform(pdist(pos))
    rows, cols = np.where((dist_mat < scaled_cutoff) & (dist_mat > 0))
    unique_mask = rows < cols
    rows = rows[unique_mask]
    cols = cols[unique_mask]
    
    bond_lengths = dist_mat[rows, cols]
    strains = np.abs(bond_lengths - 1.0)
    
    # 3. FILTER & COLOR SETUP
    # Threshold: Top 15% most stressed bonds
    threshold = np.percentile(strains, 85)
    mask = strains > threshold
    
    active_rows = rows[mask]
    active_cols = cols[mask]
    active_strains = strains[mask]
    
    # Normalize colors to the range of ACTIVE strains only
    # This ensures we use the full color spectrum
    norm = colors.Normalize(vmin=threshold, vmax=np.max(active_strains))
    # 'plasma' goes from Purple (lower stress) to Bright Yellow (max stress)
    # 'spring' goes from Magenta to Yellow (also good for dark background)
    cmap = plt.cm.plasma 
    
    # 4. PLOT PARTICLES (Very Faint Background)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='white', alpha=0.03, s=2, linewidth=0)
    
    # 5. PLOT FORCE CHAINS
    limit = 4000 # Increased limit for more detail
    count = 0
    
    # Vectorize plotting? Matplotlib 3D is slow with individual lines.
    # We will loop but optimize visuals.
    for i in range(min(len(active_rows), limit)):
        p1 = pos[active_rows[i]]
        p2 = pos[active_cols[i]]
        s = active_strains[i]
        
        # Color mapping based on strain intensity
        color = cmap(norm(s))
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                c=color, linewidth=1.2, alpha=0.9) 
                # High alpha (0.9) for sharpness

    ax.set_title(title, color='white', fontsize=14, fontweight='bold')
    ax.set_axis_off()

# Execute Plotting
fig = plt.figure(figsize=(20, 10)) # Larger Canvas

ax1 = fig.add_subplot(121, projection='3d')
plot_strain_network(ax1, raw_data[fast_idx], "Fast Glass (Chaos)\nFragmented Stress")

ax2 = fig.add_subplot(122, projection='3d')
plot_strain_network(ax2, raw_data[slow_idx], "Slow Glass (Memory)\nConnected Force Chains")

plt.tight_layout()
# DPI 300 is Print Quality (removes blur)
plt.savefig('glass_force_chains_high_res.png', dpi=300, facecolor='black') 
print("Visualization saved to 'glass_force_chains_high_res.png'.")
print("Look for the 'Filaments' in the Slow Glass.")
plt.show()