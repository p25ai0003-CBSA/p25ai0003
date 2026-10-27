# step3_bagging.py
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
SEED = d["seed"]
cfg = json.load(open("frozen_tree.json"))
y_train = d["y_train"]
n, K = len(y_train), 40
B_MAX = 200
B_LIST = [1, 5, 10, 25, 50, 100, 200]

rng = np.random.default_rng(SEED + 1)
sets = {"train": d["X_train"], "val": d["X_val"],
        "test": d["X_test"], "masked": d["X_test_masked"]}
preds = {k: np.zeros((B_MAX, len(X)), dtype=int) for k, X in sets.items()}
in_bag = np.zeros((B_MAX, n), dtype=bool)
distinct = np.zeros(B_MAX)

# ---- bagging, written by hand ----
t0 = time.time()
for b in range(B_MAX):
    idx = rng.integers(0, n, size=n)            # bootstrap: n draws WITH replacement
    in_bag[b, idx] = True
    distinct[b] = len(np.unique(idx))
    tree = DecisionTreeClassifier(max_depth=cfg["max_depth"],
                                  min_samples_leaf=cfg["min_samples_leaf"],
                                  random_state=SEED + b)
    tree.fit(d["X_train"][idx], y_train[idx])
    for k, X in sets.items():
        preds[k][b] = tree.predict(X)
print(f"fitted {B_MAX} trees in {time.time()-t0:.1f}s")

def vote_counts(p, B):                          # p: (B_MAX, N) predicted labels
    counts = np.zeros((p.shape[1], K))
    for b in range(B):
        counts[np.arange(p.shape[1]), p[b]] += 1
    return counts

def oob(B):
    counts = np.zeros((n, K))
    for b in range(B):
        out = ~in_bag[b]                        # photos this tree never saw
        counts[np.where(out)[0], preds["train"][b][out]] += 1
    has = counts.sum(1) > 0                     # photos with at least one OOB vote
    return float((counts.argmax(1)[has] == y_train[has]).mean()), float(has.mean())

def disagreement(counts, B):
    if B < 2:
        return float("nan")
    agree_pairs = (counts * (counts - 1) / 2).sum(1)
    return float((1 - agree_pairs / (B * (B - 1) / 2)).mean())

def margin(counts, B):
    f = np.sort(counts / B, axis=1)
    return float((f[:, -1] - f[:, -2]).mean())

expected = n * (1 - (1 - 1 / n) ** n)
rows = []
for B in B_LIST:
    acc = {k: float((vote_counts(preds[k], B).argmax(1) == y).mean())
           for k, y in [("train", y_train), ("val", d["y_val"]), ("test", d["y_test"])]}
    cv = vote_counts(preds["val"], B)
    oacc, cover = oob(B)
    rows.append(dict(B=B, **acc, distinct=float(distinct[:B].mean()),
                     oob=oacc, oob_cover=cover,
                     disagree=disagreement(cv, B), margin=margin(cv, B)))

print(f"\nTheoretical distinct per bootstrap: {expected:.3f}")
print(f"{'B':>4} {'train':>6} {'val':>6} {'test':>6} {'distinct':>8} {'OOB':>6} "
      f"{'cover':>6} {'disagree':>8} {'margin':>6}")
for r in rows:
    print(f"{r['B']:>4} {r['train']:6.3f} {r['val']:6.3f} {r['test']:6.3f} "
          f"{r['distinct']:8.2f} {r['oob']:6.3f} {r['oob_cover']:6.2f} "
          f"{r['disagree']:8.4f} {r['margin']:6.3f}")

# choose B by VALIDATION only; ties -> smaller B (rule fixed in advance)
best = sorted(rows, key=lambda r: (-r["val"], r["B"]))[0]
print("\nFINAL ENSEMBLE SIZE:", best["B"])

json.dump(dict(final_B=best["B"], expected_distinct=expected, rows=rows),
          open("bagging_summary.json", "w"), indent=1)
np.savez("bagging_preds.npz", in_bag=in_bag,
         **{f"preds_{k}": v for k, v in preds.items()})

# ---- plots for Q1(c) ----
Bs = [r["B"] for r in rows]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for a, key, title in zip(ax, ["oob", "disagree", "margin"],
                         ["OOB accuracy", "Pairwise disagreement", "Mean vote margin"]):
    a.plot(Bs, [r[key] for r in rows], "o-")
    a.set_xscale("log"); a.set_xlabel("Number of trees B"); a.set_title(title)
plt.tight_layout(); plt.savefig("fig_q1c.png", dpi=200)
print("saved fig_q1c.png")