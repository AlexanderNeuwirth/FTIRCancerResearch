import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sys
sys.path.insert(1, './stimlib/python')
import glob
import classify
import envi
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from PIL import Image

"""
=====================
Classifier comparison
=====================

Generation of classification map from scikit-learn classifiers on FTIR data.

Code source: Alexander Neuwirth and Sebastian Berisha
"""

# Obtained from header
BAND_1650 = 86

# Full Core 1
#X1, Y1 = 250, 100
#X2, Y2 = 1500, 1300

HALF_X = 20291 // 2
# TMA (testing half)
X1, Y1 = HALF_X + 1, 0
X2, Y2 = 20291, 17911

IMAGE_PATH='ov-63-hd-16ca-c-mnf18-bc-n'

TRAIN_MASK_PATH='/data/berisha_lab/neuwirth/annotations-ov-63-hd-16ca-c-mnf18-bc-0.05/train/'
TEST_MASK_PATH='/data/berisha_lab/neuwirth/annotations-ov-63-hd-16ca-c-mnf18-bc-0.05/test/'


classifier = RandomForestClassifier()

# load train data
data_path = '/data/berisha_lab/neuwirth/data/'
masks_path = TRAIN_MASK_PATH

classimages = sorted(glob.glob(masks_path + '*.png'))  # load the class file names
print(classimages)
exit()
C = classify.filenames2class(classimages)  # generate the class images for training

# open the ENVI file for reading, use the validation image for batch access
Etrain = envi.envi(data_path + IMAGE_PATH)

# samples = x, lines = y (in pixels). Masks are (height, width) - (y,x)
size = (Etrain.header.lines, Etrain.header.samples)
x_train, y_train = Etrain.loadtrain_balance(C, num_samples=10000)
print(f"Creating mask, size: {size}")
region_mask = np.zeros(size)
region_mask[Y1:Y2, X1:X2] = 1
print(f"Mask complete")
mappixels = Etrain.loadmask(region_mask)
Etrain.close()

name = "Random Forest"
classifier.fit(x_train, y_train)
inband = mappixels[BAND_1650, :]
inarr = inband.reshape((Y2-Y1, X2-X1))
inimg = (inarr * 255).astype(np.uint8)

print(f"dtype: {inarr.dtype} shape: {inarr.shape}")
im = Image.fromarray(inimg)
im.save("loadmap.png")
print('\n=========================================')
print(name, ' train accuracy: ', accuracy_score(y_train, classifier.predict(x_train)))

prob_map = classifier.predict_proba(mappixels.T)

# Reshape from (samples, classes) to (classes, Y, X)
prob_map = prob_map.T.reshape(-1, Y2-Y1, X2-X1)
print(f"Reshaped: {prob_map.shape}")
print(prob_map)

class_map = classify.prob2class(prob_map)
class_img = classify.class2color(class_map)
im = Image.fromarray(class_img)
im.save("classmap.png")
