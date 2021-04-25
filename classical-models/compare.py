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

"""
=====================
Classifier comparison
=====================

A comparison of several classifiers from scikit-learn on FTIR data.

Code source: Sebastian Berisha
"""

IMAGE_PATH='/data/berisha_lab/neuwirth/data/ov-63-hd-16ca-c'

TRAIN_MASK_PATH='/data/berisha_lab/neuwirth/annotations_4/train/'
TEST_MASK_PATH='/data/berisha_lab/neuwirth/annotations_4/test/'

names = ["Linear SVM"]#["Random Forest", "RBF SVM", "Linear SVM", "Neural Net", "AdaBoost",
        #"Naive Bayes", "QDA", "Decision Tree", "Nearest Neighbors"]#["Nearest Neighbors", "Linear SVM", "RBF SVM",
        # "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        # "Naive Bayes", "QDA"]
classifiers = [SVC(kernel="linear", C=0.025)]#[RandomForestClassifier(n_estimators=70, max_depth=5),
    #SVC(gamma=2, C=1), SVC(kernel="linear", C=0.025), MLPClassifier(alpha=1), AdaBoostClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis(), DecisionTreeClassifier(max_depth=10), KNeighborsClassifier(3)]

"""
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=100, max_features=16),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
"""


# iterate over classifiers
for trial in range(3):
    # load train data
    masks_path = TRAIN_MASK_PATH

    classimages = sorted(glob.glob(masks_path + '*.png'))  # load the class file names
    C = classify.filenames2class(classimages)  # generate the class images for training

    num_classes = C.shape[0]
    class_counts = {}
    for i in range(0, num_classes):
        class_counts[classimages[i]] = np.count_nonzero(C[i, :, :])

    print("Train samples:")
    print(class_counts)

    # open the ENVI file for reading, use the validation image for batch access
    Etrain = envi.envi(IMAGE_PATH)
    # x_train, y_train = Etrain.loadtrain(C)
    x_train, y_train = Etrain.loadtrain_balance(C, num_samples=10000)
    print(x_train)
    Etrain.close()

    # load test data
    masks_path = TEST_MASK_PATH

    classimages = sorted(glob.glob(masks_path + '*.png'))   # load the class file names
    print(classimages)
    C = classify.filenames2class(classimages)   # generate the class images for testing
    C = C.astype(np.uint32)

    bool_mask = np.sum(C.astype(np.uint32), 0)

    # get number of classes
    num_classes = C.shape[0]
    class_counts = {}
    for i in range(1, num_classes):
        C[i, :, :] *= i+1
    for i in range(0, num_classes):
        class_counts[classimages[i]] = np.count_nonzero(C[i, :, :])
    print("Test samples:")
    print(class_counts)

    total_mask = np.sum(C.astype(np.uint32), 0)  # validation mask

    test_set = envi.envi(IMAGE_PATH, mask=total_mask)

    N = np.count_nonzero(total_mask)  # set the batch size
    Tv = []  # initialize the target array to empty
    x_test = test_set.loadbatch(N)
    y_test = total_mask.flat[np.flatnonzero(total_mask)]  # get the indices of valid pixels
    for name, clf in zip(names, classifiers):
        clf.fit(x_train, y_train)
        print(f"Run {trial}")
        print('\n=========================================')
        print(name, ' train accuracy: ', accuracy_score(y_train, clf.predict(x_train)))
        test_predictions = clf.predict(x_test.transpose())
        print(name, ' test accuracy: ', accuracy_score(y_test, test_predictions))
        print(name, ' confusion matrix \n', confusion_matrix(y_test, test_predictions))

