"""Model.

Created: 2017-12
Last changes: 2018-01-24
Contributors: charley
"""
from __future__ import division, print_function

from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K


def downsampling_block(input_tensor, filters, padding='valid',
                       batchnorm=False, dropout=0.0):
    _, height, width, _ = K.int_shape(input_tensor)
    assert height % 2 == 0
    assert width % 2 == 0

    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(input_tensor)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return MaxPooling2D(pool_size=(2, 2))(x), x


def upsampling_block(input_tensor, skip_tensor, filters, padding='valid', batchnorm=False, dropout=0.0):
    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2))(input_tensor)

    # compute amount of cropping needed for skip_tensor
    _, x_height, x_width, _ = K.int_shape(x)
    _, s_height, s_width, _ = K.int_shape(skip_tensor)
    h_crop = s_height - x_height
    w_crop = s_width - x_width
    assert h_crop >= 0
    assert w_crop >= 0
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    else:
        cropping = ((h_crop // 2, h_crop - h_crop // 2), (w_crop // 2, w_crop - w_crop // 2))
        y = Cropping2D(cropping=cropping)(skip_tensor)

    x = Concatenate()([x, y])

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return x


def nn_architecture(height, width, channels=1, classes=1, features=32, depth=2, temperature=1.0, padding='same', batchnorm=False, dropout=0.0):
    x = Input(shape=(height, width, channels))
    inputs = x

    skips = []
    for i in range(depth):
        x, x0 = downsampling_block(x, features, padding,
                                   batchnorm, dropout)
        skips.append(x0)
        features *= 2

    x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    for i in reversed(range(depth)):
        features //= 2
        x = upsampling_block(x, skips[i], features, padding,
                             batchnorm, dropout)

    x = Conv2D(filters=classes, kernel_size=(1, 1))(x)

    logits = Lambda(lambda z: z / temperature)(x)
    probabilities = Activation('sigmoid')(logits)

    return Model(inputs=inputs, outputs=probabilities)
