'''
    # CNN model for FTIR data.
'''
from __future__ import division, print_function, absolute_import
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import initializers
import numpy as np
import os
import time

def build_net(X, Y, num_classes, num_epochs, checkpoint_path, size_batch, Xval=None, Yval=None, dec_step=100,
              train=True):

    nor = initializers.RandomNormal(stddev=0.02, seed=100)

    ###########################################################################
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X.shape[1:], kernel_initializer=nor, kernel_regularizer=regularizers.l2(0.001) ))
    #model.add(BatchNormalization(epsilon=1e-05, momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    ##########################################################################
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=nor, kernel_regularizer=regularizers.l2(0.001) ))
    #model.add(BatchNormalization(epsilon=1e-05, momentum=0.9))
    model.add(Activation('relu')) 
    
    ###########################################################################
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=nor, kernel_regularizer=regularizers.l2(0.001) ))
    #model.add(BatchNormalization(epsilon=1e-05, momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
         
    ###########################################################################
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer=nor, kernel_regularizer=regularizers.l2(0.001)))
    #model.add(BatchNormalization(epsilon=1e-05, momentum=0.9))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    ##########################################################################
    model.add(Dense(num_classes, kernel_initializer=nor))
    model.add(Activation('softmax'))

    ##########################################################################
    # initiate AdaDelta optimizer
    opt = keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-07)

    ##########################################################################
    # Let's train the model using Adadelta
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
 
    ##########################################################################
    # Train
    filepath = f"{checkpoint_path}/model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
    if train:
        start_time = time.time()
        history_callback = None
        if Xval is None or Yval is None:
            history_callback = model.fit(X, Y,
              batch_size=size_batch,
              epochs=num_epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=[checkpoint]
              )
        else:
            history_callback = model.fit(X, Y,
              batch_size=size_batch,
              epochs=num_epochs,
              validation_data=(Xval, Yval),
              shuffle=True,
              callbacks=[checkpoint]
              )

        loss_history = history_callback.history["loss"]
        loss_history_np = np.array(loss_history)
        np.savetxt("loss_history.txt", loss_history_np, delimiter=",")

        print("\n\n-------------train time: %s seconds\n\n" % (time.time() - start_time))

    return model
