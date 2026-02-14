import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("--- GLASS INVESTIGATION UNIT ---")
print("Loading data...")
with open('glass_challenge_data.pkl', 'rb') as f:
    raw_data, raw_labels = pickle.load(f)

# Arrays to hold "Dumb" features (Global Averages)
mean_stiffness = []
std_stiffness = []
density_proxy = [] # Average number of neighbors
labels = []

cutoff = 2.5 # LJ Cutoff

print("Extracting 'Dumb' Features...")
for i, item in enumerate(raw_data):
    # 1. Sanitize Features
    feats = item['features'] # (N, 3)
    if np.isnan(feats).any() or np.isinf(feats).any():
        feats = np.nan_to_num(feats, posinf=100.0, neginf=-100.0)
    
    # Feature 1: Global Mean Stiffness (Scalar)
    # If the "Slow" glass is just "Harder", this will catch it.
    mean_stiffness.append(np.mean(feats))
    
    # Feature 2: Stiffness Variance
    std_stiffness.append(np.std(feats))
    
    # Feature 3: Density / Coordination Number
    # How many neighbors does the average particle have?
    pos = item['positions']
    # Simple neighbor count approx
    from scipy.spatial.distance import pdist
    dists = pdist(pos)
    num_neighbors = np.sum(dists < cutoff)
    avg_neighbors = num_neighbors / 256.0 # 256 particles
    density_proxy.append(avg_neighbors)
    
    labels.append(raw_labels[i])

# Convert to numpy
X_dumb = np.column_stack([mean_stiffness, std_stiffness, density_proxy])
y = np.array(labels)

# --- INVESTIGATION 1: The Histogram Test ---
print("\n[Test 1] Separability Check")
print(f"Fast Glass (0) Mean Stiffness: {np.mean(X_dumb[y==0, 0]):.4f}")
print(f"Slow Glass (1) Mean Stiffness: {np.mean(X_dumb[y==1, 0]):.4f}")

# --- INVESTIGATION 2: The "Dumb Model" Test ---
# Can a simple Logistic Regression solve this?
X_train, X_test, y_train, y_test = train_test_split(X_dumb, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)
dumb_pred = clf.predict(X_test)
dumb_acc = accuracy_score(y_test, dumb_pred)

print("\n------------------------------------------------")
print(f"DUMB MODEL ACCURACY: {dumb_acc * 100:.2f}%")
print("------------------------------------------------")
print("Interpretation:")
if dumb_acc > 0.85:
    print("FAIL: The problem is TRIVIAL.")
    print("The 'Slow' glass is simply 'Stiffer' on average.")
    print("You don't need a GNN. You just need a thermometer.")
    print("Reason: The 'Annealing' process allowed particles to fall into deeper energy wells,")
    print("which mathematically guarantees higher Hessian eigenvalues.")
else:
    print("PASS: The problem is HARD.")
    print("The averages are identical. The GNN is actually looking at topology.")

# --- INVESTIGATION 3: Feature Importance ---
print("\n[Test 3] What gave it away?")
print(f"Weight on Mean Stiffness: {clf.coef_[0][0]:.4f}")
print(f"Weight on Std Stiffness:  {clf.coef_[0][1]:.4f}")
print(f"Weight on Density:        {clf.coef_[0][2]:.4f}")