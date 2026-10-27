# common.py
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

HERE = Path(__file__).parent

def load_data():
    faces = fetch_olivetti_faces()
    images, labels = faces.images, faces.target
    s = np.load(HERE / "split.npz")
    tr, va, te = s["train"], s["val"], s["test"]
    flat = lambda a: a.reshape(len(a), -1)

    masked = images[te].copy()
    masked[:, 24:40, 24:40] = 0          # hide the centre 16x16 block

    return dict(
        X_train=flat(images[tr]), y_train=labels[tr],
        X_val=flat(images[va]),   y_val=labels[va],
        X_test=flat(images[te]),  y_test=labels[te],
        X_test_masked=flat(masked),
        seed=int(s["seed"]),
    )