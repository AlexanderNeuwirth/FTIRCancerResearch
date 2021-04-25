'''
    # InceptionNet CNN model for FTIR data.
    # https://maelfabien.github.io/deeplearning/inception/#in-keras
'''
from __future__ import division, print_function, absolute_import
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from datetime import datetime
import tensorflow
import os
import numpy as np
import time

def build_net(X, Y, num_classes, num_epochs, checkpoint_path, size_batch, Xval=None, Yval=None, dec_step=100,
              train=True):

    nor = initializers.RandomNormal(stddev=0.02, seed=100)

    input_img = Input(shape=(32, 32, 16))

    ### 1st layer
    layer_1 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
    layer_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer_1)

    layer_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
    layer_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer_2)

    layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    layer_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer_3)

    mid_1 = tensorflow.keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)
    flat_1 = Flatten()(mid_1)

    dense_1 = Dense(1200, activation='relu')(flat_1)
    dense_2 = Dense(600, activation='relu')(dense_1)
    dense_3 = Dense(150, activation='relu')(dense_2)
    output = Dense(num_classes, activation='softmax')(dense_3)
    model = Model([input_img], output)
    ##########################################################################
    # initiate optimizer
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

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
    model.save(os.path.join(checkpoint_path, 'model.h5'))
    return model
