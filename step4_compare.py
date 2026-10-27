# step4_compare.py
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from common import load_data

d = load_data()
SEED, K = d["seed"], 40
cfg = json.load(open("frozen_tree.json"))
B = json.load(open("bagging_summary.json"))["final_B"]
P = np.load("bagging_preds.npz")

tree = DecisionTreeClassifier(max_depth=cfg["max_depth"],
                              min_samples_leaf=cfg["min_samples_leaf"],
                              random_state=SEED).fit(d["X_train"], d["y_train"])

def vote_counts(p, B):
    counts = np.zeros((p.shape[1], K))
    for b in range(B):
        counts[np.arange(p.shape[1]), p[b]] += 1
    return counts

y = d["y_test"]
res = {}
for name, X, key in [("clean", d["X_test"], "preds_test"),
                     ("masked", d["X_test_masked"], "preds_masked")]:
    t_pred = tree.predict(X)
    t_conf = tree.predict_proba(X).max(1)           # biggest leaf probability
    counts = vote_counts(P[key], B)
    b_pred = counts.argmax(1)
    b_conf = (counts / B).max(1)                    # biggest vote fraction
    res[name] = (t_pred, b_pred, counts)
    print(f"{name:7s} tree: acc={np.mean(t_pred==y):.4f} conf={t_conf.mean():.4f} | "
          f"bagging(B={B}): acc={np.mean(b_pred==y):.4f} conf={b_conf.mean():.4f}")

t_pred, b_pred, counts = res["clean"]
corrected = np.where((t_pred != y) & (b_pred == y))[0]
both_wrong = np.where((t_pred != y) & (b_pred != y))[0]
corrected = corrected[np.argsort(-counts[corrected, y[corrected]])]
lead = counts.max(1) - counts[np.arange(len(y)), y]
both_wrong = both_wrong[np.argsort(-lead[both_wrong])]
print("\nbagging corrects the tree on test faces:", corrected.tolist())
print("both wrong on test faces:", both_wrong.tolist())

cases = [("corrected", i) for i in corrected[:2]] + [("both wrong", i) for i in both_wrong[:2]]
fig, ax = plt.subplots(len(cases), 2, figsize=(7, 2.2 * len(cases)),
                       gridspec_kw={"width_ratios": [1, 3]})
for r, (kind, i) in enumerate(cases):
    top = np.argsort(-counts[i])[:3]
    print(f"\n[{kind}] test face {i}: true={y[i]}, tree={t_pred[i]}, bagging={b_pred[i]}, "
          f"votes(true)={int(counts[i, y[i]])}, top3 votes="
          f"{[(int(k), int(counts[i, k])) for k in top]}")
    ax[r, 0].imshow(d["X_test"][i].reshape(64, 64), cmap="gray"); ax[r, 0].axis("off")
    ax[r, 0].set_title(f"face {i}, true {y[i]}", fontsize=8)
    colors = ["gray"] * K
    colors[y[i]] = "tab:blue"
    if b_pred[i] != y[i]:
        colors[b_pred[i]] = "tab:red"
    ax[r, 1].bar(range(K), counts[i], color=colors)
    ax[r, 1].set_title(f"{kind}: tree={t_pred[i]}, bagging={b_pred[i]}", fontsize=8)
plt.tight_layout(); plt.savefig("fig_q1d.png", dpi=200)
print("\nsaved fig_q1d.png")