# step8_bias_variance.py
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from common import load_data

K = 40
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"   # smoke test, not for the report
R = 2 if QUICK else 30          # refits
B = 10 if QUICK else 100        # trees per bagged ensemble

def leaf_stats(t):
    tr = t.tree_
    leaf = tr.children_left == -1
    cnt = tr.n_node_samples[leaf]
    return int(leaf.sum()), float(cnt.mean()), float((cnt == 1).mean())

def resample_idx(rng, y):       # 6 draws with replacement inside each identity
    return np.concatenate([rng.choice(np.where(y == c)[0], size=(y == c).sum(), replace=True)
                           for c in range(K)])

def samme_q(X, y, Xte, depth, T, seed):          # same rule as steps 6 and 7
    n = len(y)
    w = np.full(n, 1 / n)
    S = np.zeros((len(Xte), K)); asum = 0.0
    for t in range(T):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=seed + t)
        tree.fit(X, y, sample_weight=w)
        miss = tree.predict(X) != y
        eps = w[miss].sum()
        if eps >= 1 - 1 / K - 1e-12:             # not better than random: reject, stop
            break
        alpha = np.log((1 - max(eps, 1e-12)) / max(eps, 1e-12)) + np.log(K - 1)
        w = w * np.exp(alpha * miss); w = w / w.sum()
        S[np.arange(len(Xte)), tree.predict(Xte)] += alpha
        asum += alpha
        if eps <= 1e-12:                         # perfect learner: stop
            break
    return S / asum if asum > 0 else np.full((len(Xte), K), 1 / K)

def one_refit(r, X_tr, y_tr, X_te, msl, depths, B, d_boost, T, seed0):
    rng = np.random.default_rng([seed0, r])
    idx = resample_idx(rng, y_tr)
    Xr, yr = X_tr[idx], y_tr[idx]
    n = len(yr)
    boots = rng.integers(0, n, size=(B, n))      # same bootstrap draws at every depth
    base = seed0 + 1000 * (r + 1)
    out = dict(idx=idx, tree={}, bag={}, leaf={}, disagree={})
    for dep in depths:
        k = str(dep)
        t = DecisionTreeClassifier(max_depth=dep, min_samples_leaf=msl,
                                   random_state=base).fit(Xr, yr)
        out["tree"][k] = t.predict_proba(X_te)   # leaf probability vector
        out["leaf"][k] = leaf_stats(t)
        C = np.zeros((len(X_te), K))
        for b in range(B):
            tb = DecisionTreeClassifier(max_depth=dep, min_samples_leaf=msl,
                                        random_state=base + 1 + b)
            tb.fit(Xr[boots[b]], yr[boots[b]])
            C[np.arange(len(X_te)), tb.predict(X_te)] += 1
        out["bag"][k] = C / B                    # vote-fraction vector
        out["disagree"][k] = float((1 - (C * (C - 1) / 2).sum(1) / (B * (B - 1) / 2)).mean())
    out["boost"] = samme_q(Xr, yr, X_te, d_boost, T, base + 500)   # normalised weighted vote
    return out

def bias_var(P, E):                              # P: (R, N, K)
    pbar = P.mean(0)
    bias2 = float(((pbar - E) ** 2).sum(1).mean())
    var = float(((P - pbar) ** 2).sum(2).mean())
    total = float(((P - E) ** 2).sum(2).mean())  # must equal bias2 + var
    return bias2, var, total

def main():
    d = load_data()
    SEED = d["seed"]
    cfg = json.load(open("frozen_tree.json"))
    BAG_B = json.load(open("bagging_summary.json"))["final_B"]
    bs = json.load(open("boost_summary.json"))
    assert cfg["max_depth"] is None, "script assumes the selected tree is unbounded"
    assert QUICK or BAG_B == B, "final bagging size must equal B"
    T = 10 if QUICK else bs["T"]
    depths = [4, 16, None]
    keys = [str(x) for x in depths]
    X_tr, y_tr, X_te, y_te = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    E = np.eye(K)[y_te]
    tag = "q3_quick" if QUICK else "q3"
    if QUICK:
        print("QUICK mode: R=2, B=10. These numbers are NOT for the report.")

    t0 = time.time()
    res = Parallel(n_jobs=int(os.environ.get("N_JOBS", -1)), verbose=5)(
        delayed(one_refit)(r, X_tr, y_tr, X_te, cfg["min_samples_leaf"], depths, B,
                           bs["depth"], T, SEED) for r in range(R))
    print(f"all {R} refits done in {(time.time() - t0) / 60:.1f} min")

    P = {("tree", k): np.stack([o["tree"][k] for o in res]) for k in keys}
    P.update({("bag", k): np.stack([o["bag"][k] for o in res]) for k in keys})
    P[("boost", "-")] = np.stack([o["boost"] for o in res])
    stats, worst = {}, 0.0
    for key, A in P.items():
        b2, v, tot = bias_var(A, E)
        acc = (A.argmax(2) == y_te).mean(1)
        worst = max(worst, abs(tot - (b2 + v)))
        stats[key] = dict(bias2=b2, var=v, acc_mean=float(acc.mean()),
                          acc_std=float(acc.std(ddof=1)))
    print(f"check: max |mean||p-e||^2 - (bias2+var)| = {worst:.2e}  (should be ~0)")

    def row(name, s):
        return (f"{name:22s} bias2={s['bias2']:.6f} var={s['var']:.6f} "
                f"acc={100*s['acc_mean']:.2f} +/- {100*s['acc_std']:.2f}")
    print(f"\nQ3(a): selected models, R={R} refits")
    print(row("tree (selected)", stats[("tree", "None")]))
    print(row(f"bagging ({B} trees)", stats[("bag", "None")]))
    print(row(f"boosting (d{bs['depth']}x{T})", stats[("boost", "-")]))

    print(f"\nQ3(b): depth study, min_samples_leaf={cfg['min_samples_leaf']}")
    print(f"{'depth':>5} {'model':6s} {'bias2':>9} {'var':>9} {'acc %':>15}")
    for k in keys:
        for m in ["tree", "bag"]:
            s = stats[(m, k)]
            print(f"{k:>5} {m:6s} {s['bias2']:9.6f} {s['var']:9.6f} "
                  f"{100*s['acc_mean']:6.2f} +/- {100*s['acc_std']:5.2f}")

    print("\nQ3(c): leaf occupancy, disagreement, variance removed, accuracy gain")
    print(f"{'depth':>5} {'leaves':>7} {'mean/leaf':>9} {'single':>6} {'disagree':>8} "
          f"{'var cut':>8} {'var cut %':>9} {'gain pts':>8}")
    for k in keys:
        lv = np.array([o["leaf"][k] for o in res])
        dis = np.mean([o["disagree"][k] for o in res])
        t, b = stats[("tree", k)], stats[("bag", k)]
        print(f"{k:>5} {lv[:,0].mean():7.1f} {lv[:,1].mean():9.2f} {lv[:,2].mean():6.3f} "
              f"{dis:8.4f} {t['var']-b['var']:8.4f} {100*(1-b['var']/t['var']):9.2f} "
              f"{100*(b['acc_mean']-t['acc_mean']):8.2f}")

    # ---- Q3(d): frozen clean/masked accuracy from saved predictions ----
    tree0 = DecisionTreeClassifier(max_depth=cfg["max_depth"],
                                   min_samples_leaf=cfg["min_samples_leaf"],
                                   random_state=SEED).fit(X_tr, y_tr)
    bp, bo = np.load("bagging_preds.npz"), np.load("boost_selected.npz")
    def bag_acc(key):
        Pm = bp[key][:BAG_B]; C = np.zeros((Pm.shape[1], K))
        for t in range(len(Pm)):
            C[np.arange(Pm.shape[1]), Pm[t]] += 1
        return float((C.argmax(1) == y_te).mean())
    def boost_acc(key):
        Pm, al = bo[key], bo["alphas"]; S = np.zeros((Pm.shape[1], K))
        for t in range(len(al)):
            S[np.arange(Pm.shape[1]), Pm[t]] += al[t]
        return float((S.argmax(1) == y_te).mean())
    final = {"tree": (tree0.score(X_te, y_te), tree0.score(d["X_test_masked"], y_te), ("tree", "None")),
             "bagging": (bag_acc("preds_test"), bag_acc("preds_masked"), ("bag", "None")),
             "boosting": (boost_acc("preds_test"), boost_acc("preds_masked"), ("boost", "-"))}
    print("\nQ3(d): final table")
    print(f"{'model':9s} {'clean %':>8} {'masked %':>9} {'bias2':>9} {'var':>9}")
    for name, (c, m, key) in final.items():
        s = stats[key]
        print(f"{name:9s} {100*c:8.2f} {100*m:9.2f} {s['bias2']:9.6f} {s['var']:9.6f}")

    json.dump({f"{a}|{b}": s for (a, b), s in stats.items()}, open(f"{tag}_stats.json", "w"), indent=1)
    np.savez_compressed(f"{tag}_vectors.npz", resample_idx=np.stack([o["idx"] for o in res]),
                        **{f"{a}_{b}": A for (a, b), A in P.items()})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    xs = np.arange(len(keys))
    for a, f, title in zip(ax, ["bias2", "var", "acc_mean"],
                           ["Bias squared", "Variance", "Mean test accuracy"]):
        for m, lab in [("tree", "single tree"), ("bag", f"bagging ({B})")]:
            a.plot(xs, [stats[(m, k)][f] for k in keys], "o-", label=lab)
        a.set_xticks(xs); a.set_xticklabels(["4", "16", "unbounded"])
        a.set_xlabel("max depth"); a.set_title(title)
    ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"fig_{tag}_depth.png", dpi=200)
    print(f"\nsaved {tag}_stats.json, {tag}_vectors.npz, fig_{tag}_depth.png")

if __name__ == "__main__":
    main()