from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.preprocessing.image as im

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

base_dir = os.path.join('data', 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

BATCH_SIZE = 128
IMG_SHAPE = 224

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


image_gen = im.ImageDataGenerator(rescale=1./255,
                              rotation_range=45,
                              zoom_range=.5,
                              horizontal_flip=True,
                              width_shift_range=.15,
                              height_shift_range=.15)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                              directory=train_dir,
                                              shuffle=True,
                                              target_size=(IMG_SHAPE, IMG_SHAPE),
                                              class_mode='sparse')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = im.ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                directory=val_dir,
                                                shuffle=False,
                                                target_size=(IMG_SHAPE, IMG_SHAPE),
                                                class_mode='sparse')


class MobileNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.Conv_32 = Conv2D(32, (3, 3), padding='same', strides=2, activation='linear')
        self.Conv_64 = Conv2D(64, (3, 3), padding='same', strides=1, activation='linear')
        self.Conv_128 = Conv2D(128, (3, 3), padding='same', strides=1, activation='linear')
        self.Conv_256 = Conv2D(256, (3, 3), padding='same', strides=1, activation='linear')
        self.Conv_512 = Conv2D(512, (3, 3), padding='same', strides=1, activation='linear')
        self.Conv_1024 = Conv2D(1024, (3, 3), padding='same', strides=1, activation='linear')
        self.DepthConv_1 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='linear',
                                           depth_multiplier=1,
                                           depthwise_regularizer='l2', bias_regularizer='l2')
        self.DepthConv_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same', activation='linear',
                                           depth_multiplier=1,
                                           depthwise_regularizer='l2', bias_regularizer='l2')

        self.relu = ReLU()
        self.BN = BatchNormalization()
        self.flatten = Flatten()
        self.pool = GlobalAveragePooling2D()
        self.Dense = Dense(units=5, activation='softmax')

    def call(self, inputs, training=True):
        h = self.Conv_32(inputs)
        h = self.BN(h, training=training)
        h = self.relu(h)

        h = self.DepthConv_1(h)
        h = self.BN(h, training=training)
        h = self.relu(h)
        h = self.Conv_64(h)
        h = self.BN(h, training=training)
        h = self.relu(h)

        h = self.DepthConv_2(h)
        h = self.BN(h, training=training)
        h = self.relu(h)
        h = self.Conv_128(h)
        h = self.BN(h, training=training)
        h = self.relu(h)

        h = self.DepthConv_1(h)
        h = self.BN(h, training=training)
        h = self.relu(h)
        h = self.Conv_128(h)
        h = self.BN(h, training=training)
        h = self.relu(h)

        h = self.DepthConv_2(h)
        h = self.BN(h, training=training)
        h = self.relu(h)
        h = self.Conv_256(h)
        h = self.BN(h, training=training)
        h = self.relu(h)

        h = self.DepthConv_1(h)
        h = self.BN(h, training=training)
        h = self.relu(h)
        h = self.Conv_256(h)
        h = self.BN(h, training=training)
        h = self.relu(h)

        h = self.DepthConv_2(h)
        h = self.BN(h, training=training)
        h = self.relu(h)
        h = self.Conv_512(h)
        h = self.BN(h, training=training)
        h = self.relu(h)

        for i in range(5):
            h = self.DepthConv_1(h)
            h = self.BN(h, training=training)
            h = self.relu(h)
            h = self.Conv_512(h)
            h = self.BN(h, training=training)
            h = self.relu(h)

        for i in range(2):
            h = self.DepthConv_2(h)
            h = self.BN(h, training=training)
            h = self.relu(h)
            h = self.Conv_1024(h)
            h = self.BN(h, training=training)
            h = self.relu(h)

        h = self.pool(h)
        h = self.flatten(h)
        h = self.Dense(h)
        return h

model = MobileNet()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# for i in range(10):
for batch_index in range(train_data_gen.n // BATCH_SIZE):
    print(batch_index)
    with tf.GradientTape() as tape:
        y_pred = model(train_data_gen[batch_index][0], training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=train_data_gen[batch_index][1], y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch {}: loss {}".format(batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))