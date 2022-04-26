import os

import numpy as np
import tqdm as tqdm
from tensorflow.keras.optimizers import Adam
from keras_unet.models import vanilla_unet
from matplotlib import pyplot as plt
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
    X, y = unet.get_data(unet.path)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

    # input_img = Input((unet.im_height, unet.im_width, 1), name='i (1)')

    model = vanilla_unet(input_shape=(512, 512, 3))

    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                        validation_data=(X_valid, y_valid))

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
             label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();