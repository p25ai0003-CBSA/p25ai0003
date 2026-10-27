# step1_split.py
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from pathlib import Path
Here = Path(__file__).parent

SEED = 250003                      # digits of roll number, in order
rng = np.random.default_rng(SEED)

faces = fetch_olivetti_faces()     # needs internet the first time
images = faces.images              # shape (400, 64, 64)
labels = faces.target              # shape (400,), values 0..39

train_idx, val_idx, test_idx = [], [], []
for person in range(40):
    idx = np.where(labels == person)[0]        # this person's 10 photos
    shuffled = rng.permutation(idx)            # shuffle them
    train_idx += list(shuffled[:6])
    val_idx   += list(shuffled[6:8])
    test_idx  += list(shuffled[8:10])

train_idx, val_idx, test_idx = map(np.array, (train_idx, val_idx, test_idx))

# save the split so every later step uses exactly the same piles
np.savez(Here / "split.npz", train=train_idx, val=val_idx, test=test_idx, seed=SEED)

def flat(imgs):
    return imgs.reshape(len(imgs), -1)         # 64x64 -> 4096 numbers

# masked copy of the test images (rows/cols 24..39 -> slice 24:40)
test_masked_imgs = images[test_idx].copy()
test_masked_imgs[:, 24:40, 24:40] = 0

X_train, y_train = flat(images[train_idx]), labels[train_idx]
X_val,   y_val   = flat(images[val_idx]),   labels[val_idx]
X_test,  y_test  = flat(images[test_idx]),  labels[test_idx]
X_test_masked    = flat(test_masked_imgs)

print("train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)
print("masked test:", X_test_masked.shape)
print("zeros in masked block:", (test_masked_imgs[:, 24:40, 24:40] == 0).all())
print("overlap between piles:",
      len(set(train_idx) & set(val_idx)), len(set(train_idx) & set(test_idx)),
      len(set(val_idx) & set(test_idx)))