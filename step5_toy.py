# step5_toy.py
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=float)
y = np.array([0, 0, 1, 0, 1, 1, 2, 2, 2])
K, n = 3, len(y)

w = np.full(n, 1 / n)                                    # equal weights to start
stump = DecisionTreeClassifier(max_depth=1, random_state=0)
stump.fit(X, y, sample_weight=w)
pred = stump.predict(X)
miss = pred != y

eps = w[miss].sum()
print("random-guess error threshold 1-1/K =", round(1 - 1 / K, 4), "| eps =", round(eps, 4))
assert eps < 1 - 1 / K, "weak learner not better than random"

alpha = np.log((1 - eps) / eps) + np.log(K - 1)
w_raw = w * np.exp(alpha * miss)
w_new = w_raw / w_raw.sum()

print(f"alpha = log((1-eps)/eps) + log(K-1) = {alpha:.5f}")
print(f"{'i':>2} {'x':>3} {'y':>2} {'pred':>4} {'w_old':>8} {'w_raw':>8} {'w_new':>8}")
for i in range(n):
    print(f"{i:>2} {X[i,0]:>3.0f} {y[i]:>2} {pred[i]:>4} {w[i]:8.4f} {w_raw[i]:8.4f} {w_new[i]:8.4f}")
print("sum of new weights:", w_new.sum())