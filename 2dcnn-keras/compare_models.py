import argparse
import alexnet_model
import vgg16_model
import inception_model
import optir_model
import utils_keras
import os
import numpy as np
import spectral.io.envi as envi
import time
import scipy.misc
import sys
sys.path.insert(1, './stimlib/python')
import classify
from tensorflow.keras.models import load_model


################ read command line arguments####################################################
parser = argparse.ArgumentParser(description="Train a CNN model on FTIR HSI data -- \
                                             the input samples are cropped patches around each pixel")

#required args
required = parser.add_argument_group('required named arguments')
required.add_argument("--data", help="Path to train data folder.", type=str)
required.add_argument("--masks", help="Path to train masks folder.", type=str)
required.add_argument("--test_data", help="Path to train data folder.", type=str)
required.add_argument("--test_masks", help="Path to train masks folder.", type=str)
required.add_argument("--checkpoint", help="Path to checkpoint directory.", type=str)
required.add_argument("--crop", help="Crop size", type=int)
required.add_argument("--classes", help="Num of classes.", type=int)
required.add_argument("--epochs", help="Num of epochs to use for training.", type=int)
required.add_argument("--batch", help="Batch size to use for training.", type=int)

# optional arguments
parser.add_argument("--balance", help="Balance number of training samples for each class", action="store_true")
parser.add_argument("--samples", help="Num of train samples per class.", type=int)
parser.add_argument("--validate", help="validate on new set", action="store_true")
parser.add_argument("--valdata", help="Path to validation data folder.", type=str)
parser.add_argument("--valmasks", help="Path to validation masks folder.", type=str)
parser.add_argument("--valbalance", help="Balance number of validation samples for each class.", action="store_true")
parser.add_argument("--valsamples", help="Num of validation samples per class.", type=int)

args = parser.parse_args()

print('\n.............loading training data...........\n')
X,Y, num_bands = utils_keras.load_data(args.data, args.masks, args.crop, args.classes, samples=args.samples, balance=args.balance)
print('\n..............done loading training data.........\n')

###############################################################################
# load new validation set
###############################################################################
if args.validate:
    # load validation data from different envi file
    print('\n...........loading validation data............\n')
    X_val,Y_val, num_bands = utils_keras.load_data(args.valdata, args.valmasks, args.crop, args.classes, samples=args.valsamples, balance=args.valbalance)
    print('\n...........done loading valdiation data......\n')
else:
    # validate on a subset of training data
    X_val = None
    Y_val = None

models = [inception_model, vgg16_model]

for model in models:
    #utils_keras.chp_folder(args.checkpoint) # delete contents of checkpoint folder if it exists
    print(f'============ training with {model.__name__}==========\n')
    model = model.build_net(X,
        Y,
        args.classes,
        args.epochs,
        args.checkpoint,
        args.batch,
        Xval=X_val,
        Yval=Y_val)
    print('\n ======= classifying =========\n')

                                    
    model = load_model(os.path.join(args.checkpoint, 'model.h5'))

    total_samples = 0
    print('\n\n ... computing cnn overall accuracy \n\n')
    probs, conf_mat = utils_keras.cnn_metrics(args.test_data, args.test_masks, args.crop, args.classes, model)
    print('\n Confusion matrix \n', conf_mat)
