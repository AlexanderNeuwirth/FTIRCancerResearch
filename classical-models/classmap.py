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

# All tissue
#X1, Y1 = (12700, 14950)
#X2, Y2 = (12750, 15000)

# All zero
#X1, Y1 = (20000,6000)
#X2, Y2 = (20100,6100)

# Full Core 1
#X1, Y1 = 250, 100
#X2, Y2 = 1500, 1300

HALF_X = 20291 // 2
# All cores
X1, Y1 = 0, 0
X2, Y2 = HALF_X, 17911
#X1, Y1 = HALF_X + 1, 0
#X2, Y2 = 20291, 17911

# Mix
#X1, Y1 = (500, 100)
#X2, Y2 = (700, 550)#(1500, 1300)

IMAGE_PATH='/data/berisha_lab/neuwirth/data/mnf16/ov-63-hd-16ca-c-mnf16-bc-n-pca16'

TRAIN_MASK_PATH='/data/berisha_lab/neuwirth/annotations_3/train/'


classifier = GaussianNB()#RandomForestClassifier()

# load train data
masks_path = TRAIN_MASK_PATH

classimages = sorted(glob.glob(masks_path + '*.png'))  # load the class file names
C = classify.filenames2class(classimages)  # generate the class images for training

# open the ENVI file for reading, use the validation image for batch access
Etrain = envi.envi(IMAGE_PATH)

# samples = x, lines = y (in pixels). Masks are (height, width) - (y,x)
size = (Etrain.header.lines, Etrain.header.samples)
# x_train, y_train = Etrain.loadtrain(C)
x_train, y_train = Etrain.loadtrain_balance(C, num_samples=10000)
print(f"Creating mask, size: {size}")
region_mask = np.zeros(size)
region_mask[Y1:Y2, X1:X2] = 1
print(f"Mask complete")
mappixels = Etrain.loadmask(region_mask)
Etrain.close()

"""
# load test data
data_path = '/data/berisha_lab/neuwirth/data/'
masks_path = TEST_MASK_PATH

classimages = sorted(glob.glob(masks_path + '*.png'))   # load the class file names
print(classimages)
C = classify.filenames2class(classimages)   # generate the class images for testing
C = C.astype(np.uint32)

bool_mask = np.sum(C.astype(np.uint32), 0)
# get number of classes
num_classes = C.shape[0]

for i in range(1, num_classes):
    C[i, :, :] *= i+1

total_mask = np.sum(C.astype(np.uint32), 0)  # validation mask

test_set = envi.envi(data_path + IMAGE_PATH, mask=total_mask)

N = np.count_nonzero(total_mask)  # set the batch size
Tv = []  # initialize the target array to empty
x_test = test_set.loadbatch(N)
y_test = total_mask.flat[np.flatnonzero(total_mask)]  # get the indices of valid pixels
"""
name = "Gaussian"#"Random Forest"
classifier.fit(x_train, y_train)
#inband = mappixels[BAND_1650, :]
#inarr = inband.reshape((Y2-Y1, X2-X1))
#inimg = (inarr * 255).astype(np.uint8)

#print(f"dtype: {inarr.dtype} shape: {inarr.shape}")
#im = Image.fromarray(inimg)
#im.save("loadmap.png")
print('\n=========================================')
print(name, ' train accuracy: ', accuracy_score(y_train, classifier.predict(x_train)))

prob_map = classifier.predict_proba(mappixels.T)#y_train.transpose())
print(prob_map.shape)
print(prob_map)
# Reshape from (samples, classes) to (classes, Y, X)
prob_map = prob_map.T.reshape(-1, Y2-Y1, X2-X1)
print(f"Reshaped: {prob_map.shape}")
print(prob_map)

class_map = classify.prob2class(prob_map)
class_img = classify.class2color(class_map)
im = Image.fromarray(class_img)
im.save("classmap_train_gaussian.png")

#print(name, ' test accuracy: ', accuracy_score(y_test, test_predictions))
#print(name, ' confusion matrix \n', confusion_matrix(y_test, test_predictions))

