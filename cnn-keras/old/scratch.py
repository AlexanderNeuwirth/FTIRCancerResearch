import argparse
import hsi_cnn_model_keras
import utils_keras
import os

checkpoint = '/Users/msoesdl/Box/research/optir/checkpoints'

utils_keras.chp_folder(checkpoint)

data = '/Users/msoesdl/Box/research/optir/data/row-c-bkg/envi/'
masks = '/Users/msoesdl/Box/research/optir/data/row-c-bkg/train/'
crop = 17
classes = 2
samples = 1000

print('\n.............loading training data...........\n')
X,Y, num_bands = utils_keras.load_data(data, masks, crop, classes, samples)
print('\n..............done loading training data.........\n')

X_val = None
Y_val = None

print('============ training with hsi_cnn_model==========\n')
print('X shape: ', X.shape)
print('Y shape: ', Y.shape)

epochs=8
batch=128
model = hsi_cnn_model_keras.build_net(X,
                                Y,
                                classes,
                                epochs,
                                checkpoint,
                                batch,
                                Xval=X_val,
                                Yval=Y_val)

model.save(os.path.join(checkpoint, 'model.h5'))

print('\n\t Training done.')

#%%
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
#%%
temp = imageio.imread('rows-c-j-bkg-train-stroma.png', as_gray=True)
temp = np.divide(temp, np.amax(temp))

#%%
idx = np.flatnonzero(temp)
samples = 10000
np.random.shuffle(idx)
idx = idx[0:samples]

#%%
t = np.zeros(temp.shape)
t.flat[idx] = 255

#%%
imageio.imwrite('./test.png', t.astype(np.uint8))

#%%
plt.imshow(t), plt.show()

#%%
samples = 100000
t = np.zeros(temp.shape)
idx = np.transpose(np.nonzero(temp))
np.random.shuffle(idx)
idx = idx[0:samples, :]
for (r, c) in idx:
    t[r,c] = 255

imageio.imwrite('new-test.png', t.astype(np.uint8))