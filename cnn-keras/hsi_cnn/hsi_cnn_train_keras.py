import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import argparse
import cnn3d_model
import optir_model
import utils_keras
import hsi_cnn_model_keras_012921
import hsi_cnn_model_keras_relu
import os

################ read command line arguments####################################################
parser = argparse.ArgumentParser(description="Train a CNN model on FTIR HSI data -- \
                                             the input samples are cropped patches around each pixel")

#required args
required = parser.add_argument_group('required named arguments')
required.add_argument("--data", help="Path to train data folder.", type=str)
required.add_argument("--masks", help="Path to train masks folder.", type=str)
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

checkpoint_path = f"/data/berisha_lab/neuwirth/code/2dcnn-keras/checkpoint/{args.checkpoint}"

utils_keras.chp_folder(checkpoint_path)   # delete contents of checkpoint folder if it exists

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


# Real-time data preprocessing
#img_prep = ImagePreprocessing()
#img_prep.add_featurewise_zero_center()
#img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
#img_aug = ImageAugmentation()

# set up network


print(f'============ training {args.checkpoint}==========\n')
print('X shape: ', X.shape)
print('Y shape: ', Y.shape)
if X_val is not None and Y_val is not None:
    print('Xval shape: ', X_val.shape)
    print('Yval shape: ', Y_val.shape)

model = hsi_cnn_model_keras_012921.build_net(X,
    Y,
    args.classes,
    args.epochs,
    checkpoint_path,
    args.batch,
    Xval=X_val,
    Yval=Y_val)

print('\n\t Training done.')
