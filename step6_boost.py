# step6_boost.py
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from common import load_data

d = load_data()
SEED, K = d["seed"], 40
y_tr = d["y_train"]
n = len(y_tr)
sets = {"train": d["X_train"], "val": d["X_val"], "test": d["X_test"],
        "masked": d["X_test_masked"]}
ys = {"train": y_tr, "val": d["y_val"], "test": d["y_test"], "masked": d["y_test"]}

def samme_fit(depth, T):
    w = np.full(n, 1 / n)
    alphas, preds = [], {k: [] for k in sets}
    log = dict(eps=[], alpha=[], maxw=[], top10=[], w=[], correct=[])
    for t in range(T):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED + t)
        tree.fit(sets["train"], y_tr, sample_weight=w)
        p_tr = tree.predict(sets["train"])
        miss = p_tr != y_tr
        eps = w[miss].sum()
        if eps >= 1 - 1 / K - 1e-12:                     # not better than random: reject
            if t == 0:
                raise RuntimeError("first weak learner is not better than random")
            print(f"  depth {depth}: stopped at round {t}, eps={eps:.4f}")
            break
        alpha = np.log((1 - max(eps, 1e-12)) / max(eps, 1e-12)) + np.log(K - 1)
        w = w * np.exp(alpha * miss)
        w = w / w.sum()
        alphas.append(alpha)
        for k, X in sets.items():
            preds[k].append(tree.predict(X))
        log["eps"].append(eps); log["alpha"].append(alpha)
        log["maxw"].append(w.max())
        log["top10"].append(np.sort(w)[-10:].sum())
        log["w"].append(w.copy()); log["correct"].append(~miss)
        if eps <= 1e-12:                                 # perfect learner: stop
            print(f"  depth {depth}: perfect learner at round {t}, stopping")
            break
    return np.array(alphas), {k: np.array(v) for k, v in preds.items()}, log

def scores(alphas, P, T):
    S = np.zeros((P.shape[1], K))
    for t in range(min(T, len(alphas))):
        S[np.arange(P.shape[1]), P[t]] += alphas[t]
    return S / alphas[:min(T, len(alphas))].sum()        # q_k(x)

grid, fits = [], {}
t0 = time.time()
for depth in [1, 2, 3, 4]:
    s = time.time()
    fits[depth] = samme_fit(depth, 100)
    print(f"depth {depth} fitted in {time.time()-s:.1f}s")
    al, P, _ = fits[depth]
    for T in [10, 25, 50, 100]:
        acc = {k: float((scores(al, P[k], T).argmax(1) == ys[k]).mean())
               for k in ["train", "val", "test"]}
        grid.append(dict(depth=depth, T=T, **acc))

print(f"\n{'depth':>5} {'rounds':>6} {'train':>6} {'val':>6}")
for g in grid:
    print(f"{g['depth']:>5} {g['T']:>6} {g['train']:6.3f} {g['val']:6.3f}")

# choose by VALIDATION only; ties -> smaller depth, then fewer rounds (fixed in advance)
best = sorted(grid, key=lambda g: (-g["val"], g["depth"], g["T"]))[0]
print("\nSELECTED:", best)

al, P, log = fits[best["depth"]]
T = best["T"]
json.dump(dict(depth=best["depth"], T=T, grid=grid), open("boost_summary.json", "w"), indent=1)
np.savez("boost_selected.npz", alphas=al[:T],
         **{f"preds_{k}": v[:T] for k, v in P.items()},
         W=np.array(log["w"][:T]), correct=np.array(log["correct"][:T]))

# ---- Q2(b) plots ----
r = np.arange(1, T + 1)
fig, ax = plt.subplots(1, 4, figsize=(15, 3))
for a, key, title in zip(ax, ["eps", "alpha", "maxw", "top10"],
        ["Weighted training error", "Learner weight alpha", "Max sample weight",
         "Weight on 10 heaviest images"]):
    a.plot(r, log[key][:T]); a.set_title(title); a.set_xlabel("Round")
ax[0].axhline(1 - 1 / K, color="r", ls="--")
plt.tight_layout(); plt.savefig("fig_q2b.png", dpi=200)

# ---- heaviest training face (for Q2c) ----
W = np.array(log["w"][:T]); corr = np.array(log["correct"][:T])
j = int(W.max(0).argmax())
print(f"\nHeaviest training face: train position {j}, identity {y_tr[j]}")
print(f"  max weight {W[:, j].max():.4f} (uniform start = {1/n:.4f})")
print(f"  weak learners got it wrong in {int((~corr[:, j]).sum())} of {T} rounds")
print("  correct-by-round:", corr[:, j].astype(int).tolist())
print("saved fig_q2b.png, boost_selected.npz")