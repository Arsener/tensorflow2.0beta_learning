from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import *


class MobileNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = Conv2D(32, (3, 3), padding='same', strides=2, activation='linear')
        self.bn = BatchNormalization()

        self.depth_1 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='linear',
                                       depth_multiplier=1)
        self.bn_1_1 = BatchNormalization()
        self.conv_1 = Conv2D(64, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_1_2 = BatchNormalization()

        self.depth_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same', activation='linear',
                                       depth_multiplier=1)
        self.bn_2_1 = BatchNormalization()
        self.conv_2 = Conv2D(128, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_2_2 = BatchNormalization()

        self.depth_3 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='linear',
                                       depth_multiplier=1)
        self.bn_3_1 = BatchNormalization()
        self.conv_3 = Conv2D(128, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_3_2 = BatchNormalization()

        self.depth_4 = DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same', activation='linear',
                                       depth_multiplier=1)
        self.bn_4_1 = BatchNormalization()
        self.conv_4 = Conv2D(256, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_4_2 = BatchNormalization()

        self.depth_5 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='linear',
                                       depth_multiplier=1)
        self.bn_5_1 = BatchNormalization()
        self.conv_5 = Conv2D(256, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_5_2 = BatchNormalization()

        self.depth_6 = DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same', activation='linear',
                                       depth_multiplier=1)
        self.bn_6_1 = BatchNormalization()
        self.conv_6 = Conv2D(512, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_6_2 = BatchNormalization()

        self.depth_7 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='linear',
                                       depth_multiplier=1)
        self.bn_7_1 = BatchNormalization()
        self.conv_7 = Conv2D(512, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_7_2 = BatchNormalization()

        self.depth_8 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='linear',
                                       depth_multiplier=1)
        self.bn_8_1 = BatchNormalization()
        self.conv_8 = Conv2D(512, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_8_2 = BatchNormalization()

        self.depth_9 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='linear',
                                       depth_multiplier=1)
        self.bn_9_1 = BatchNormalization()
        self.conv_9 = Conv2D(512, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_9_2 = BatchNormalization()

        self.depth_10 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='linear',
                                        depth_multiplier=1)
        self.bn_10_1 = BatchNormalization()
        self.conv_10 = Conv2D(512, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_10_2 = BatchNormalization()

        self.depth_11 = DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='linear',
                                        depth_multiplier=1)
        self.bn_11_1 = BatchNormalization()
        self.conv_11 = Conv2D(512, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_11_2 = BatchNormalization()

        self.depth_12 = DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same', activation='linear',
                                        depth_multiplier=1)
        self.bn_12_1 = BatchNormalization()
        self.conv_12 = Conv2D(1024, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_12_2 = BatchNormalization()

        self.depth_13 = DepthwiseConv2D(kernel_size=(3, 3), strides=2, padding='same', activation='linear',
                                        depth_multiplier=1)
        self.bn_13_1 = BatchNormalization()
        self.conv_13 = Conv2D(1024, (1, 1), padding='same', strides=1, activation='linear')
        self.bn_13_2 = BatchNormalization()

        self.relu = ReLU()
        self.flatten = Flatten()
        self.pool = GlobalAveragePooling2D()
        self.Dense = Dense(units=5, activation='softmax')

    def call(self, inputs, training=True):
        h = self.conv(inputs)
        h = self.bn(h, training=training)
        h = self.relu(h)

        h = self.depth_1(h)
        h = self.bn_1_1(h, training=training)
        h = self.relu(h)
        h = self.conv_1(h)
        h = self.bn_1_2(h, training=training)
        h = self.relu(h)

        h = self.depth_2(h)
        h = self.bn_2_1(h, training=training)
        h = self.relu(h)
        h = self.conv_2(h)
        h = self.bn_2_2(h, training=training)
        h = self.relu(h)

        h = self.depth_3(h)
        h = self.bn_3_1(h, training=training)
        h = self.relu(h)
        h = self.conv_3(h)
        h = self.bn_3_2(h, training=training)
        h = self.relu(h)

        h = self.depth_4(h)
        h = self.bn_4_1(h, training=training)
        h = self.relu(h)
        h = self.conv_4(h)
        h = self.bn_4_2(h, training=training)
        h = self.relu(h)

        h = self.depth_5(h)
        h = self.bn_5_1(h, training=training)
        h = self.relu(h)
        h = self.conv_5(h)
        h = self.bn_5_2(h, training=training)
        h = self.relu(h)

        h = self.depth_6(h)
        h = self.bn_6_1(h, training=training)
        h = self.relu(h)
        h = self.conv_6(h)
        h = self.bn_6_2(h, training=training)
        h = self.relu(h)

        h = self.depth_7(h)
        h = self.bn_7_1(h, training=training)
        h = self.relu(h)
        h = self.conv_7(h)
        h = self.bn_7_2(h, training=training)
        h = self.relu(h)

        h = self.depth_8(h)
        h = self.bn_8_1(h, training=training)
        h = self.relu(h)
        h = self.conv_8(h)
        h = self.bn_8_2(h, training=training)
        h = self.relu(h)

        h = self.depth_9(h)
        h = self.bn_9_1(h, training=training)
        h = self.relu(h)
        h = self.conv_9(h)
        h = self.bn_9_2(h, training=training)
        h = self.relu(h)

        h = self.depth_10(h)
        h = self.bn_10_1(h, training=training)
        h = self.relu(h)
        h = self.conv_10(h)
        h = self.bn_10_2(h, training=training)
        h = self.relu(h)

        h = self.depth_11(h)
        h = self.bn_11_1(h, training=training)
        h = self.relu(h)
        h = self.conv_11(h)
        h = self.bn_11_2(h, training=training)
        h = self.relu(h)

        h = self.depth_12(h)
        h = self.bn_12_1(h, training=training)
        h = self.relu(h)
        h = self.conv_12(h)
        h = self.bn_12_2(h, training=training)
        h = self.relu(h)

        h = self.depth_13(h)
        h = self.bn_13_1(h, training=training)
        h = self.relu(h)
        h = self.conv_13(h)
        h = self.bn_13_2(h, training=training)
        h = self.relu(h)

        h = self.pool(h)
        h = self.flatten(h)
        h = self.Dense(h)
        return h
