import os

import tensorflow as tf
from keras.preprocessing.image_dataset import image_dataset_from_directory
from keras_unet.models import vanilla_unet

if __name__ == '__main__':


    batch_size = 1
    image_size = (512, 512)
    train_dataset = image_dataset_from_directory('D:/gits/Unettestproj/source/proceedimgdir',
                                                 subset='training',
                                                 seed=42,
                                                 validation_split=0.1,
                                                 batch_size=batch_size,
                                                 image_size=image_size)

    validation_dataset = image_dataset_from_directory('D:/gits/Unettestproj/source/proceedimgdir',
                                                      subset='validation',
                                                      seed=42,
                                                      validation_split=0.1,
                                                      batch_size=batch_size,
                                                      image_size=image_size)

    test_dataset = image_dataset_from_directory('D:/gits/Unettestproj/source/proceedimgdir',
                                                batch_size=batch_size,
                                                image_size=image_size)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    model = vanilla_unet(input_shape=(512, 512, 3))

    model.compile()

    print(model.summary())

