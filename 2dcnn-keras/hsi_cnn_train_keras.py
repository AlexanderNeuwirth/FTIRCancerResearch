import argparse
import hsi_cnn_model_keras_012921
import hsi_cnn_model_keras
import optir_model
import utils_keras
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

utils_keras.chp_folder(args.checkpoint)   # delete contents of checkpoint folder if it exists

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


print('============ training with hsi_cnn_model_keras_012921==========\n')
print('X shape: ', X.shape)
print('Y shape: ', Y.shape)
if X_val is not None and Y_val is not None:
    print('Xval shape: ', X_val.shape)
    print('Yval shape: ', Y_val.shape)

# Normalize using sklearn 
'''
num_train_samples = X.shape[0]

X = np.reshape(X, (X.shape[0]*X.shape[1]*X.shape[2], X.shape[3])) 
X = X.astype(float)
print('\n\n X reshape: ', X.shape)

scaler = preprocessing.StandardScaler().fit(X)

X = scaler.transform(X)

print('\n\n X_scaled shape: ', X.shape)

print('\n\n Mean of X_scaled: ', X.mean(axis=0))

print('\n\n Std of X_scaled: ', X.std(axis=0))


scaler_filename = "scaler_file"
joblib.dump(scaler, scaler_filename)

if X_val is not None:
    X_val = X_val.astype(float)
    num_val_samples = X_val.shape[0]
    X_val = scaler.transform(np.reshape(X_val, (X_val.shape[0]*X_val.shape[1]*X_val.shape[2], X_val.shape[3])))
    X_val = np.reshape(X_val, (num_val_samples, args.crop, args.crop, 16))

print('\n\n Mean of X_val: ', X_val.mean(axis=0))

print('\n\n Std of X_val: ', X_val.std(axis=0))


# Reshape to satisfy data format by keras
# Last dim set to 16 manually
X = np.reshape(X, (num_train_samples, args.crop, args.crop, 16))


print('\n\n X_scaled shape: ', X.shape)
print('\n\n X_val_scaled shape: ', X_val.shape)
'''
#model = optir_model.build_net(X,
model = hsi_cnn_model_keras_012921.build_net(X,
    Y,
    args.classes,
    args.epochs,
    args.checkpoint,
    args.batch,
    Xval=X_val,
    Yval=Y_val)


#model.save(os.path.join(args.checkpoint, 'model.h5'))

print('\n\t Training done.')
