import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
z = np.load("boost_selected.npz"); W, c, a = z["W"], z["correct"], z["alphas"]
Wprev = np.vstack([np.full(W.shape[1], 1 / W.shape[1]), W[:-1]])
eps = (Wprev * ~c).sum(1)
print(f"eps   min {eps.min():.3f}  max {eps.max():.3f}  mean {eps.mean():.3f}")
print(f"alpha min {a.min():.3f}  max {a.max():.3f}")
print("check alpha formula, max abs diff:", np.abs(np.log((1-eps)/eps)+np.log(39)-a).max())
print(f"max weight {W.max():.4f}; top-10 weight at last round {np.sort(W[-1])[-10:].sum():.3f}")
