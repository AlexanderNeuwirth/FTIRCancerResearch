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

    #Instantiation
    AlexNet = Sequential()

    #1st Convolutional Layer
    AlexNet.add(Conv2D(filters=96, input_shape=(32,32,16), kernel_size=(11,11), strides=(4,4), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #2nd Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #3rd Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #4th Convolutional Layer
    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    #5th Convolutional Layer
    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    #Passing it to a Fully Connected layer
    AlexNet.add(Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(Dropout(0.4))

    #2nd Fully Connected Layer
    AlexNet.add(Dense(4096))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #3rd Fully Connected Layer
    AlexNet.add(Dense(1000))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    #Add Dropout
    AlexNet.add(Dropout(0.4))

    #Output Layer
    AlexNet.add(Dense(num_classes))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('softmax'))


    ##########################################################################
    # initiate optimizer
    AlexNet.compile(loss = keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])

    ##########################################################################
    model = AlexNet
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
