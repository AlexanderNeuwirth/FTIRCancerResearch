"""

3D CNN model for FTIR data.
Taken from "A Fast and Compact 3-D CNN for
Hyperspectral Image Classification", Ahmad Et al.
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9307220

"""
from __future__ import division, print_function, absolute_import

import keras
from keras.layers import Dropout, Input, Conv2D, Conv3D, MaxPool3D, Flatten, Dense, Reshape, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import tensorflow as tf
import os
import numpy as np
import time

def build_net(X, Y, num_classes, num_epochs, checkpoint_path, size_batch, Xval=None, Yval=None, dec_step=100,
              train=True):

    """Build a 3D convolutional neural network model."""
    
    X = X[:, : , :, :, np.newaxis]
    Xval = Xval[:, : , :, :, np.newaxis]
    input_layer = Input((11, 11, 16, 1))
    ## 3D Convolutional Layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
    ## Flatten 3D Convolutional Layer
    flatten_layer = Flatten()(conv_layer4)
    ## Fully Connected Layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=num_classes, activation='softmax')(dense_layer2)

    # define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    adam = Adam(lr=0.001, decay=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    filepath = f"{checkpoint_path}/model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Train
    if train:
        start_time = time.time()
        history_callback = None
        if Xval is None or Yval is None:
            history_callback = model.fit(X, Y,
              batch_size=size_batch,
              epochs=num_epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=[checkpoint, tensorboard_callback])
        else:
            history_callback = model.fit(X, Y,
              batch_size=size_batch,
              epochs=num_epochs,
              validation_data=(Xval, Yval),
              shuffle=True,
              callbacks=[checkpoint, tensorboard_callback])

        loss_history = history_callback.history["loss"]
        loss_history_np = np.array(loss_history)
        np.savetxt("loss_history.txt", loss_history_np, delimiter=",")

        print("\n\n-------------train time: %s seconds\n\n" % (time.time() - start_time))

    return model
