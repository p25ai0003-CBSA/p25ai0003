# step7_ensembles.py
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
y_tr = d["y_train"]; n = len(y_tr)
cfg = json.load(open("frozen_tree.json"))
BAG_B = json.load(open("bagging_summary.json"))["final_B"]
bs = json.load(open("boost_summary.json"))
D_BOOST, T_BOOST = bs["depth"], bs["T"]
sets = {"train": d["X_train"], "val": d["X_val"], "test": d["X_test"],
        "masked": d["X_test_masked"]}
ys = {"train": y_tr, "val": d["y_val"], "test": d["y_test"], "masked": d["y_test"]}

def fit_bagging(B):                       # same seeds as step 3
    rng = np.random.default_rng(SEED + 1)
    preds = {k: np.zeros((B, len(X)), dtype=int) for k, X in sets.items()}
    times = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        tree = DecisionTreeClassifier(max_depth=cfg["max_depth"],
                                      min_samples_leaf=cfg["min_samples_leaf"],
                                      random_state=SEED + b)
        t0 = time.perf_counter()
        tree.fit(sets["train"][idx], y_tr[idx])
        times[b] = time.perf_counter() - t0
        for k, X in sets.items():
            preds[k][b] = tree.predict(X)
    return preds, times

def fit_samme(depth, T):                  # same rule as step 6
    w = np.full(n, 1 / n)
    alphas, times, Ws, corrs = [], [], [], []
    preds = {k: [] for k in sets}
    for t in range(T):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED + t)
        t0 = time.perf_counter()
        tree.fit(sets["train"], y_tr, sample_weight=w)
        miss = tree.predict(sets["train"]) != y_tr
        eps = w[miss].sum()
        if eps >= 1 - 1 / K - 1e-12:
            if t == 0:
                raise RuntimeError("first weak learner is not better than random")
            print(f"stopped at round {t}: eps={eps:.4f}")
            break
        alpha = np.log((1 - max(eps, 1e-12)) / max(eps, 1e-12)) + np.log(K - 1)
        w = w * np.exp(alpha * miss)
        w = w / w.sum()
        times.append(time.perf_counter() - t0)
        alphas.append(alpha); Ws.append(w.copy()); corrs.append(~miss)
        for k, X in sets.items():
            preds[k].append(tree.predict(X))
        if eps <= 1e-12:
            print(f"perfect learner at round {t}, stopping"); break
    return (np.array(alphas), {k: np.array(v) for k, v in preds.items()},
            np.array(times), np.array(Ws), np.array(corrs))

def votes(P, m):
    m = min(m, len(P))
    C = np.zeros((P.shape[1], K))
    for t in range(m):
        C[np.arange(P.shape[1]), P[t]] += 1
    return C

def scores(al, P, m):                     # q_k(x)
    m = min(m, len(al))
    S = np.zeros((P.shape[1], K))
    for t in range(m):
        S[np.arange(P.shape[1]), P[t]] += al[t]
    return S / al[:m].sum()

def pair_disagree(C, B):                  # per-face disagreement among B trees
    return 1 - (C * (C - 1) / 2).sum(1) / (B * (B - 1) / 2)

bag_P, bag_t = fit_bagging(100)
al, boo_P, boo_t, W, corr = fit_samme(D_BOOST, 100)

# ---------------- Q2(c) ----------------
print(f"\nQ2(c): equal learner counts (boosting depth {D_BOOST})")
print("CHECK: bagging rows must match step 3 (25: val .838 test .700; "
      "50: .850/.838; 100: .887/.850)")
print(f"{'model':9s} {'T':>4} {'train':>6} {'val':>6} {'test':>6} {'fit time s':>10}")
for m in [25, 50, 100]:
    a = {k: (votes(bag_P[k], m).argmax(1) == ys[k]).mean() for k in ["train", "val", "test"]}
    print(f"{'bagging':9s} {m:>4} {a['train']:6.3f} {a['val']:6.3f} {a['test']:6.3f} "
          f"{bag_t[:m].sum():10.1f}")
    a = {k: (scores(al, boo_P[k], m).argmax(1) == ys[k]).mean() for k in ["train", "val", "test"]}
    print(f"{'boosting':9s} {m:>4} {a['train']:6.3f} {a['val']:6.3f} {a['test']:6.3f} "
          f"{boo_t[:m].sum():10.1f}")

Tb = T_BOOST
j = int(W[:Tb].max(0).argmax())
hist = boo_P["train"][:Tb, j]
wrong = hist[hist != y_tr[j]]
v, c = np.unique(wrong, return_counts=True)
print(f"\nHeaviest training face: position {j}, identity {y_tr[j]}, "
      f"max weight {W[:Tb, j].max():.4f} (start {1/n:.4f})")
print(f"  wrong in {len(wrong)} of {Tb} rounds; predicted label per round:\n  {hist.tolist()}")
print("  most common wrong labels (label, count):",
      sorted(zip(v.tolist(), c.tolist()), key=lambda t: -t[1])[:5])

r = np.arange(1, Tb + 1)
fig, ax = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [1, 3]})
ax[0].imshow(d["X_train"][j].reshape(64, 64), cmap="gray"); ax[0].axis("off")
ax[0].set_title(f"train face {j}, identity {y_tr[j]}", fontsize=8)
ax[1].plot(r, W[:Tb, j])
ax[1].plot(r[corr[:Tb, j]], W[:Tb, j][corr[:Tb, j]], "ro", label="learner correct")
ax[1].axhline(1 / n, ls=":", color="k")
ax[1].set_xlabel("Boosting round"); ax[1].set_ylabel("Sample weight"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_q2c_face.png", dpi=200)

# ---------------- Q2(d) ----------------
tree = DecisionTreeClassifier(max_depth=cfg["max_depth"],
                              min_samples_leaf=cfg["min_samples_leaf"],
                              random_state=SEED).fit(sets["train"], y_tr)
y = d["y_test"]
res = {}
print(f"\nQ2(d): frozen models (tree, bagging B={BAG_B}, boosting depth {D_BOOST} x {Tb})")
for cond, X, key in [("clean", d["X_test"], "test"), ("masked", d["X_test_masked"], "masked")]:
    res[cond] = dict(tree=tree.predict_proba(X),
                     bag=votes(bag_P[key], BAG_B) / BAG_B,
                     boost=scores(al, boo_P[key], Tb))
    for name in ["tree", "bag", "boost"]:
        S = res[cond][name]
        print(f"{cond:7s} {name:6s} acc={np.mean(S.argmax(1) == y):.4f} "
              f"mean winning={S.max(1).mean():.4f}")

pred = {(c, m): res[c][m].argmax(1) for c in res for m in res[c]}
bag_same = (pred["clean", "bag"] == y) == (pred["masked", "bag"] == y)
boo_chg = (pred["clean", "boost"] == y) != (pred["masked", "boost"] == y)
cand = np.where(bag_same & boo_chg)[0]
dq = np.abs(res["masked"]["boost"][cand, y[cand]] - res["clean"]["boost"][cand, y[cand]])
cand = cand[np.argsort(-dq)]
print("\ncandidates (bagging correctness unchanged, boosting changes):", cand.tolist())
if len(cand) == 0:
    raise SystemExit("no face matches the rule; tell me and we will pick another rule")

i = int(cand[0])
Cc = votes(bag_P["test"], BAG_B); Cm = votes(bag_P["masked"], BAG_B)
print(f"\nCHOSEN test face {i}, true identity {y[i]}")
for cond, C in [("clean", Cc), ("masked", Cm)]:
    wb, qb = int(C[i].argmax()), res[cond]["boost"][i]
    print(f"  {cond:6s} bagging: winner {wb} ({int(C[i].max())}/{BAG_B} votes), "
          f"true-class votes {int(C[i, y[i]])}, tree disagreement "
          f"{pair_disagree(C, BAG_B)[i]:.4f}")
    print(f"  {cond:6s} boosting: winner {int(qb.argmax())} (q={qb.max():.4f}), "
          f"q(true)={qb[y[i]]:.4f}")
top10 = np.argsort(W[Tb - 1])[-10:]
print(f"  heaviest 10 training faces at last round carry {W[Tb-1][top10].sum():.3f} of the weight; "
      f"their identities: {sorted(y_tr[top10].tolist())}")
print(f"  is identity {y[i]} among them? {bool(y[i] in y_tr[top10])}")

fig, ax = plt.subplots(1, 4, figsize=(15, 3))
ax[0].imshow(d["X_test"][i].reshape(64, 64), cmap="gray"); ax[0].set_title(f"clean, true {y[i]}", fontsize=8)
ax[1].imshow(d["X_test_masked"][i].reshape(64, 64), cmap="gray"); ax[1].set_title("masked", fontsize=8)
ax[0].axis("off"); ax[1].axis("off")
x = np.arange(K)
for a, name, title in [(ax[2], "bag", "bagging vote fraction"), (ax[3], "boost", "boosting q_k")]:
    a.bar(x - 0.2, res["clean"][name][i], width=0.4, label="clean")
    a.bar(x + 0.2, res["masked"][name][i], width=0.4, label="masked")
    a.axvline(y[i], color="g", ls=":"); a.set_title(title, fontsize=9)
ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_q2d.png", dpi=200)
print("\nsaved fig_q2c_face.png, fig_q2d.png")