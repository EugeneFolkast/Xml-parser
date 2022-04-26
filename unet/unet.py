import os

import numpy as np
import tqdm as tqdm
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import xmlparser.xmlparser

class unet:
    def __init__(self):
        self.im_width = 512
        self.im_height = 512
        self.path = 'D:/gits/Unettestproj/source/sourceimg/'

    def get_data(self, train = True):
        ids = next(os.walk(self.path))[2]
        X = np.zeros((len(ids), self.im_height, self.im_width, 1), dtype=np.float32)

        if train:
            y = np.zeros((len(ids), self.im_height, self.im_width, 1), dtype=np.float32)

        for n, id_ in tqdm.tqdm_notebook(enumerate(ids), total=len(ids)):
            # Load images
            img = load_img(self.path + id_, color_mode = "grayscale")
            x_img = img_to_array(img)
            x_img = resize(x_img, (512, 512, 1), mode='constant', preserve_range=True)

            # Load masks
            if train:
                mask = img_to_array(load_img('D:/gits/Unettestproj/source/proceedimgdir/data/' + id_, color_mode = "grayscale"))
                mask = resize(mask, (512, 512, 1), mode='constant', preserve_range=True)

            # Save images
            X[n, ..., 0] = x_img.squeeze() / 255
            if train:
                y[n] = mask / 255
        print('Done!')

        if train:
            return X, y
        else:
            return X

if __name__ == '__main__':
    unet = unet()
    X, y = unet.get_data()

    print(X + '  ' + y)