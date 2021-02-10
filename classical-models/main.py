import numpy as np
from spectral import *
from PIL import Image
import time
from sklearn.ensemble import RandomForestClassifier

classes = ["epithelium"]#, "stroma", "necrosis", "blood", "lymphocytes"]

start = time.time()
def current(): return round(time.time() - start, 2)
def log(s): print(f"{current()} {s}")

log("Loaded imports.")

masks = {}
samples_per_class = 10000
for tissue in classes:
    for set in ["train", "test"]:
        masks[f"{tissue}_{set}"] = np.copy(np.asarray(Image.open(f"/data/berisha_lab/neuwirth/annotations-1-masks/{set}/{tissue}_{set}.png")))
        mask = masks[f"{tissue}_{set}"]
        mask[mask > 0] = 1
        x,y = np.where(mask == 1)
        selected = np.random.choice(len(x), (10000,), replace=False)
        newmask = np.zeros_like(mask)
        newmask[x[selected], y[selected]] = True
        masks[f"{tissue}_{set}"] = newmask 
        log(f"Loaded {tissue}_{set}.")

log("Finished loading masks.")

OV63 = '/data/berisha_lab/ftir/with-paraffin/ov-63/hd/16ca/ov-63-hd-16ca.hdr'
img = open_image(OV63)
log(f"Loaded imaging file.")

y_train = np.zeros_like(masks["epithelium_train"])
y_test = np.zeros_like(masks["epithelium_test"])

for i in range(len(classes)):
    label = classes[i]
    mask_test = masks[f"{label}_test"]
    mask_train = masks[f"{label}_train"]
    y_train[mask_train == 1] = i + 1
    y_test[mask_test == 1] = i + 1

log("Isolating annotated pixels...")
mask_train = y_train > 0
mask_test = y_test > 0

spectra = img.shape[2]
# Preallocate for number of annotated pixels
X_train = np.zeros((np.count_nonzero(mask_test), spectra))
log(f"Preallocated {X_train.shape}")
sample = 0
print("Nonzero:")
print(len(y_train.nonzero()))
nz = y_train.nonzero()
for i in zip(np.nditer(nz[0]), np.nditer(nz[1])):
    if sample % 1000 == 0:
        log(f"Processed data through sample {sample}")
    X_train[sample, :] = img[i]
    sample += 1


#X_train = img[y_train > 0]
#print(X_train.shape)
#X_test = img[y_test > 0]
#print(X_test.shape)

print(np.unique(X_train, return_counts=True))

clf = RandomForestClassifier(verbose=10)
log("Fitting classifier...")
clf.fit(X_train, y_train)

log("Predicting on test samples...")
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
log(f"Accuracy: {accuracy}") 


