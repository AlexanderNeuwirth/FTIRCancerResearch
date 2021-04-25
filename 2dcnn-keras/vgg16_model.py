'''
    # AlexNet CNN model for FTIR data.
'''
from __future__ import division, print_function, absolute_import
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import initializers
from datetime import datetime
import tensorflow as tf
import os
import numpy as np
import time

def build_net(X, Y, num_classes, num_epochs, checkpoint_path, size_batch, Xval=None, Yval=None, dec_step=100,
              train=True):

    nor = initializers.RandomNormal(stddev=0.02, seed=100)

    model = Sequential()
    model.add(Conv2D(input_shape=(32,32,16),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=num_classes, activation="softmax"))


    ##########################################################################
    # initiate optimizer
    from keras.optimizers import Adam
    opt = Adam(lr=0.001)
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer= opt, metrics=['accuracy'])
    model.summary()
    ##########################################################################

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    filepath = f"{checkpoint_path}/model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

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
