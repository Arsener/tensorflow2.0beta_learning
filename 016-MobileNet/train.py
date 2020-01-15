from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from MobileNet import MobileNet
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

DATASET = 'flower_photos'
BATCH_SIZE = 128
IMG_SHAPE = 224


def get_generator(dataset, train=True):
    base_path = os.path.join('..', 'data')
    base_path = os.path.join(base_path, dataset)
    path = os.path.join(base_path, 'train') if train else os.path.join(base_path, 'val')

    if train:
        image_generator = ImageDataGenerator(rescale=1. / 255,
                                             rotation_range=45,
                                             zoom_range=.2,
                                             horizontal_flip=True,
                                             width_shift_range=.15,
                                             height_shift_range=.15,
                                             fill_mode='nearest')

    else:
        image_generator = ImageDataGenerator(rescale=1. / 255)

    data_generator = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=path,
                                                         shuffle=train,
                                                         target_size=(IMG_SHAPE, IMG_SHAPE),
                                                         class_mode='sparse')

    return data_generator


if __name__ == '__main__':
    train_data_gen = get_generator(DATASET, True)
    val_data_gen = get_generator(DATASET, False)

    model = MobileNet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    EPOCHS = 1000
    train_batches = train_data_gen.n // BATCH_SIZE
    val_batches = val_data_gen.n // BATCH_SIZE

    for i in range(EPOCHS):
        print('-----EPOCH: {}-----'.format(i))
        for batch_index in range(train_batches):
            with tf.GradientTape() as tape:
                y_pred = model(train_data_gen[batch_index][0], training=True)
                y_true = train_data_gen[batch_index][1]
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)
                loss = tf.reduce_mean(loss)
                acc.update_state(y_true=y_true, y_pred=y_pred)
                print("EPOCH {} batch {}: loss {}, accuracy {}"
                      .format(i, batch_index, loss.numpy(), acc.result().numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        val_acc = 0
        for batch_index in range(val_batches):
            y_pred = model(val_data_gen[batch_index][0], training=False)
            acc.update_state(y_true=val_data_gen[batch_index][1], y_pred=y_pred)
            val_acc += acc.result().numpy()
        val_acc /= val_batches
        print("EPOCH: {}: validate accuracy {}".format(i, val_acc))

'''
0.8010083198547363

'''