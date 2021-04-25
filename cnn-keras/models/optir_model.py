'''
    # CNN model for OPTIR data.
'''
from __future__ import division, print_function, absolute_import
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras import regularizers
from keras import initializers
import numpy as np
import os
import time

def build_net(X, Y, num_classes, num_epochs, checkpoint_path, size_batch, Xval=None, Yval=None, dec_step=100,
              train=True):

    nor = initializers.RandomNormal()

    ###########################################################################
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:] ) )
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    ##########################################################################
    model.add(Conv2D(32, (3, 3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ###########################################################################
    model.add(Conv2D(32, (3, 3) ))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
         
    ###########################################################################
    #model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(32))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    ##########################################################################
    model.add(Dense(5))
    model.add(Activation('softmax'))

    ##########################################################################
    # initiate AdaDelta optimizer
    opt = keras.optimizers.Adam()

    ##########################################################################
    # Let's train the model using Adadelta
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
 
    print(model.summary())
    ##########################################################################
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
              verbose=2)
        else:
            history_callback = model.fit(X, Y,
              batch_size=size_batch,
              epochs=num_epochs,
              validation_data=(Xval, Yval),
              shuffle=True)

        loss_history = history_callback.history["loss"]
        loss_history_np = np.array(loss_history)
        np.savetxt("loss_history.txt", loss_history_np, delimiter=",")
        print("\n\n-------------train time: %s seconds\n\n" % (time.time() - start_time))

    return model
