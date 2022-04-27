import os
import time
import glob

import adam
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage import measure
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter

class Unet:
    def __init__(self):
        self.CLASSES = 14

        self.COLORS = ['black', 'red', 'lime',
                  'blue', 'orange', 'pink',
                  'cyan', 'magenta', 'crimson',
                       'deeppink', 'tomato', 'deeporange',
                       'gold', 'yellow']

        self.SAMPLE_SIZE = (256, 256)

        self.OUTPUT_SIZE = (1080, 1920)

    def load_images(self, image, mask):
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image)
        image = tf.image.resize(image, self.OUTPUT_SIZE)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image / 255.0

        mask = tf.io.read_file(mask)
        mask = tf.io.decode_png(mask)
        # mask = tf.image.rgb_to_grayscale(mask)
        mask = tf.image.resize(mask, self.OUTPUT_SIZE)
        mask = tf.image.convert_image_dtype(mask, tf.float32)

        masks = []

        for i in range(self.CLASSES):
            masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

        masks = tf.stack(masks, axis=2)
        masks = tf.reshape(masks, self.OUTPUT_SIZE + (self.CLASSES,))

        return image, masks

    def augmentate_images(self, image, masks):
        random_crop = tf.random.uniform((), 0.3, 1)
        image = tf.image.central_crop(image, random_crop)
        masks = tf.image.central_crop(masks, random_crop)

        random_flip = tf.random.uniform((), 0, 1)
        if random_flip >= 0.5:
            image = tf.image.flip_left_right(image)
            masks = tf.image.flip_left_right(masks)

        image = tf.image.resize(image, self.SAMPLE_SIZE)
        masks = tf.image.resize(masks, self.SAMPLE_SIZE)

        return image, masks

    def input_layer(self):
        return tf.keras.layers.Input(shape=self.SAMPLE_SIZE + (3,))

    def downsample_block(self, filters, size, batch_norm=True):
        initializer = tf.keras.initializers.GlorotNormal()

        result = tf.keras.Sequential()

        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if batch_norm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())
        return result

    def upsample_block(self, filters, size, dropout=False):
        initializer = tf.keras.initializers.GlorotNormal()

        result = tf.keras.Sequential()

        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                            kernel_initializer=initializer, use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if dropout:
            result.add(tf.keras.layers.Dropout(0.25))

        result.add(tf.keras.layers.ReLU())
        return result

    def output_layer(self, size):
        initializer = tf.keras.initializers.GlorotNormal()
        return tf.keras.layers.Conv2DTranspose(self.CLASSES, size, strides=2, padding='same',
                                               kernel_initializer=initializer, activation='sigmoid')

    def dice_mc_metric(self, a, b):
        a = tf.unstack(a, axis=3)
        b = tf.unstack(b, axis=3)

        dice_summ = 0

        for i, (aa, bb) in enumerate(zip(a, b)):
            numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
            denomerator = tf.math.reduce_sum(aa + bb) + 1
            dice_summ += numenator / denomerator

        avg_dice = dice_summ / self.CLASSES

        return avg_dice

    def dice_mc_loss(self, a, b):
        return 1 - self.dice_mc_metric(a, b)

    def dice_bce_mc_loss(self, a, b):
        return 0.3 * self.dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    unet = Unet()

    images = sorted(glob.glob('D:/gits/Unettestproj/source/sourceimg/*.jpg'))
    masks = sorted(glob.glob('D:/gits/Unettestproj/source/proceedimgdir/pngdata/*.png'))

    images_dataset = tf.data.Dataset.from_tensor_slices(images)
    masks_dataset = tf.data.Dataset.from_tensor_slices(masks)

    dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

    dataset = dataset.map(unet.load_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat(60)
    dataset = dataset.map(unet.augmentate_images, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = dataset.take(2000).cache()
    test_dataset = dataset.skip(2000).take(100).cache()

    train_dataset = train_dataset.batch(8)
    test_dataset = test_dataset.batch(8)

    inp_layer = unet.input_layer()

    downsample_stack = [
        unet.downsample_block(64, 4, batch_norm=False),
        unet.downsample_block(128, 4),
        unet.downsample_block(256, 4),
        unet.downsample_block(512, 4),
        unet.downsample_block(512, 4),
        unet.downsample_block(512, 4),
        unet.downsample_block(512, 4),
    ]

    upsample_stack = [
        unet.upsample_block(512, 4, dropout=True),
        unet.upsample_block(512, 4, dropout=True),
        unet.upsample_block(512, 4, dropout=True),
        unet.upsample_block(256, 4),
        unet.upsample_block(128, 4),
        unet.upsample_block(64, 4)
    ]

    out_layer = unet.output_layer(4)

    # Реализуем skip connections
    x = inp_layer

    downsample_skips = []

    for block in downsample_stack:
        x = block(x)
        downsample_skips.append(x)

    downsample_skips = reversed(downsample_skips[:-1])

    for up_block, down_block in zip(upsample_stack, downsample_skips):
        x = up_block(x)
        x = tf.keras.layers.Concatenate()([x, down_block])

    out_layer = out_layer(x)

    unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)

    tf.keras.utils.plot_model(unet_like, show_shapes=True, dpi=72)

    unet_like.compile(optimizer=tf.keras.optimizers.Adam(), loss=[unet.dice_bce_mc_loss], metrics=[unet.dice_mc_metric])

    history_dice = unet_like.fit(train_dataset, validation_data=test_dataset, epochs=25, initial_epoch=0)

    unet_like.save_weights('D:/gits/Unettestproj/source/weights/')

    unet_like.load_weights('D:/gits/Unettestproj/source/weights/')

    rgb_colors = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 165, 0),
        (255, 192, 203),
        (0, 255, 255),
        (255, 0, 255),

        (220, 20, 60),
        (255, 20, 147),
        (255, 99, 71),
        (255, 140, 0),
        (255, 215, 0),
        (255, 255, 0)


    ]

    frames = sorted(glob.glob('D:/gits/Unettestproj/source/test/*.jpg'))

    for filename in frames:
        frame = imread(filename)
        sample = resize(frame, unet.SAMPLE_SIZE)

        predict = unet_like.predict(sample.reshape((1,) + unet.SAMPLE_SIZE + (3,)))
        predict = predict.reshape(unet.SAMPLE_SIZE + (unet.CLASSES,))

        scale = frame.shape[0] / unet.SAMPLE_SIZE[0], frame.shape[1] / unet.SAMPLE_SIZE[1]

        frame = (frame / 1.5).astype(np.uint8)

        for channel in range(1, unet.CLASSES):
            contour_overlay = np.zeros((frame.shape[0], frame.shape[1]))
            contours = measure.find_contours(np.array(predict[:, :, channel]))

            try:
                for contour in contours:
                    rr, cc = polygon_perimeter(contour[:, 0] * scale[0],
                                               contour[:, 1] * scale[1],
                                               shape=contour_overlay.shape)

                    contour_overlay[rr, cc] = 1

                contour_overlay = dilation(contour_overlay, disk(1))
                frame[contour_overlay == 1] = rgb_colors[channel]
            except:
                pass

        imsave(f'D:/gits/Unettestproj/source/proceedimgdir{os.path.basename(filename)}', frame)



