import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET = 'flower_photos'
BATCH_SIZE = 128  # Number of training examples to process before updating our models variables
IMG_SHAPE = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels


def get_generator(dataset, train=True):
    base_path = os.path.join('..', 'data')
    base_path = os.path.join(base_path, dataset)
    path = os.path.join(base_path, 'train') if train else os.path.join(base_path, 'val')

    if train:
        image_generator = ImageDataGenerator(rescale=1. / 255,
                                             rotation_range=30,
                                             shear_range=0.2,
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

    if train:
        print('Total {} images in training set.'.format(data_generator.n))
    else:
        print('Total {} images in validation set.'.format(data_generator.n))
    return data_generator


def SimpleVGG_A_Net():
    inputs = tf.keras.Input(shape=(IMG_SHAPE, IMG_SHAPE, 3))
    h = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                               input_shape=(IMG_SHAPE, IMG_SHAPE, 3))(inputs)

    h = tf.keras.layers.MaxPooling2D(2, 2)(h)
    h = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(h)

    h = tf.keras.layers.MaxPooling2D(2, 2)(h)
    h = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(h)
    h = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(h)

    h = tf.keras.layers.MaxPooling2D(2, 2)(h)
    h = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(h)
    h = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(h)

    h = tf.keras.layers.MaxPooling2D(2, 2)(h)
    h = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(h)
    h = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(h)

    h = tf.keras.layers.MaxPooling2D(2, 2)(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    h = tf.keras.layers.Dense(4096, activation='relu')(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    h = tf.keras.layers.Dense(4096, activation='relu')(h)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(h)

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    train_data_gen = get_generator(DATASET, True)
    val_data_gen = get_generator(DATASET, False)

    model = SimpleVGG_A_Net()
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['sparse_categorical_accuracy'])
    model.summary()

    EPOCHS = 1000
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # 是否有提升关注的指标
            monitor='val_sparse_categorical_accuracy',
            # 不再提升的阈值
            min_delta=1e-2,
            # 20个epoch没有提升就停止
            patience=20,
            verbose=1)
    ]

    history = model.fit_generator(train_data_gen,
                                  epochs=EPOCHS,
                                  steps_per_epoch=int(np.ceil(train_data_gen.n / float(BATCH_SIZE))),
                                  validation_data=val_data_gen,
                                  validation_steps=int(np.ceil(val_data_gen.n / float(BATCH_SIZE))),
                                  callbacks=callbacks,
                                  verbose=1)

    test_loss, test_accuracy = model.evaluate(val_data_gen, steps=int(np.ceil(val_data_gen.n / float(BATCH_SIZE))))
    print('Loss and accuracy on test dataset: {0:.4f}, {1:.4f}'.format(test_loss, test_accuracy))

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='best')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Training and Validation Loss')
    plt.show()

'''
A+150：
0.7116,最高0.8136
0.7442，最高0.7646
0.7891，最高0.8122
'''
