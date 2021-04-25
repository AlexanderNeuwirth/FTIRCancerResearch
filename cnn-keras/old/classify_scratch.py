import numpy as np
import argparse
import hsi_cnn_model_keras
from scipy import io
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import time
import utils_keras
import scipy.misc
#import classify
import os
from keras.models import load_model

data = '/Users/msoesdl/Box/research/optir/data/row-c-bkg/envi/'
masks = '/Users/msoesdl/Box/research/optir/data/row-c-bkg/test/'

num_samples, num_bands, _, _ = utils_keras.envi_dims(data, masks)

print('\n ======= classifying using hsi_cnn_model =========\n')

checkpoint = '/Users/msoesdl/Box/research/optir/checkpoints'

model = load_model(os.path.join(checkpoint, 'model.h5'))

total_samples = 0
npixels=10000
metrics=False
crop = 17
classes=2

if npixels > 0 and metrics:
    print('\n\n ... classifying data using a load batch size of: ', npixels, ' pixels')
    probs, conf_mat = utils_keras.cnn_classify_batch(data, masks, crop, classes, model,
                                                     npixels)
    # save response array as envi file
    envi.save_image(metrics + '-response.hdr', probs, dtype=np.float32, interleave='bsq', force=True)

    print('\n Confusion matrix \n', conf_mat)
    np.savetxt(metrics + '-conf-mat', conf_mat, delimiter=",", fmt="%1f")
    oa = np.trace(conf_mat) / np.sum(conf_mat)
    print('\n\t==================>OA: %0.2f%%' % (oa * 100))

    # save a classified image
    #class_image = classify.prob2class(np.rollaxis(probs, 2, 0))
    #rgb = classify.class2color(class_image)
    #scipy.misc.imsave(args.metrics + '-classified-image.png', rgb)

elif npixels:
    print('\n\n Computing confusion matrix and overall accuracy')
    print('\n\n ... classifying data using a load batch size of: ', npixels, ' pixels')
    _, conf_mat = utils_keras.cnn_classify_batch(data, masks, crop, classes, model, npixels)

    print('\n Confusion matrix \n', conf_mat)
    oa = np.trace(conf_mat) / np.sum(conf_mat)
    print('\n\t==================>OA: %0.2f%%' % (oa * 100))

elif metrics:
    print('\n\n ... computing cnn metrics \n\n')
    probs, conf_mat = utils_keras.cnn_metrics(data, masks, crop, classes, model)

    # save response array as envi file
    envi.save_image(metrics + '-response.hdr', probs, dtype=np.float32, interleave='bsq', force=True)
    print('\n Confusion matrix \n', conf_mat)
    np.savetxt(metrics + '-conf-mat', conf_mat, delimiter=",", fmt="%1f")
else:
    print('\n\n ... computing cnn overall accuracy \n\n')
    # get overall accuracy
    X, Y, num_bands = utils_keras.load_data(data, masks, crop, classes)

    # X = np.rollaxis(X, 3, 2) # needed for 1d cnn

    # Evaluate model
    start_time = time.time()
    score = model.evaluate(X, Y)
    print("\n\n-------------test time: %s seconds\n\n" % (time.time() - start_time))

    print('\n\t==================>Test accuracy: %0.2f%%' % (score[0] * 100))