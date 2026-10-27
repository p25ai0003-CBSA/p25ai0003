import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_data

d = load_data()
SEED = d["seed"]

def leaf_stats(clf):
    t = clf.tree_
    is_leaf = t.children_left == -1
    counts = t.n_node_samples[is_leaf]
    return int(is_leaf.sum()), float(counts.mean()), float((counts == 1).mean())

rows = []
for depth in [2, 4, 8, 16, None]:
    for msl in [1, 2, 4]:
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl,
                                     random_state=SEED)
        clf.fit(d["X_train"], d["y_train"])
        n_leaves, mean_leaf, frac_single = leaf_stats(clf)
        rows.append(dict(
            depth=depth, msl=msl, real_depth=clf.get_depth(),
            train=clf.score(d["X_train"], d["y_train"]),
            val=clf.score(d["X_val"], d["y_val"]),
            test=clf.score(d["X_test"], d["y_test"]),
            leaves=n_leaves, mean_leaf=mean_leaf, frac_single=frac_single,
        ))

print(f"{'depth':>5} {'msl':>3} {'real':>4} {'train':>6} {'val':>6} {'test':>6} "
      f"{'leaves':>6} {'mean/leaf':>9} {'single':>6}")
for r in rows:
    print(f"{str(r['depth']):>5} {r['msl']:>3} {r['real_depth']:>4} {r['train']:6.3f} "
          f"{r['val']:6.3f} {r['test']:6.3f} {r['leaves']:6d} "
          f"{r['mean_leaf']:9.2f} {r['frac_single']:6.2f}")

best = sorted(rows, key=lambda r: (-r["val"], r["leaves"]))[0]
print("\nFROZEN SETTING:", dict(max_depth=best["depth"], min_samples_leaf=best["msl"]))
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})

with open("frozen_tree.json", "w") as f:
    json.dump(dict(max_depth=best["depth"], min_samples_leaf=best["msl"]), f)
