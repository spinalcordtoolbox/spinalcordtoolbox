# coding: utf-8
# This is the deepseg_gm model definition for the
# Spinal Cord Gray Matter Segmentation.
#
# Reference paper:
#     Perone, C. S., Calabrese, E., & Cohen-Adad, J. (2017).
#     Spinal cord gray matter segmentation using deep dilated convolutions.
#     URL: https://arxiv.org/abs/1710.01269

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, Dropout
from keras.layers import RepeatVector, Reshape
from keras.layers import BatchNormalization
from keras.layers import concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam

CROP_WIDTH = 200
CROP_HEIGHT = 200

# Models
# Tuple of (model, metadata)
MODELS = {
    'challenge': ('challenge_model.hdf5', 'challenge_model.json'),
    'large': ('large_model.hdf5', 'large_model.json'),
}


def dice_coef(y_true, y_pred):
    """Dice coefficient specification

    :param y_true: ground truth.
    :param y_pred: predictions.
    """
    dice_smooth_factor = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + dice_smooth_factor) / \
           (K.sum(y_true_f) + K.sum(y_pred_f) + dice_smooth_factor)


def dice_coef_loss(y_true, y_pred):
    """Dice loss specification.

    :param y_true: ground truth.
    :param y_pred: predictions.
    """
    return -dice_coef(y_true, y_pred)


def create_model(nfilters):
    drop_rate_concat = 0.4
    drop_rate_hidden = 0.4
    bn_momentum = 0.1

    inputs = Input((CROP_HEIGHT, CROP_WIDTH, 1))

    conv1 = Conv2D(nfilters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization(momentum=bn_momentum)(conv1)
    conv1 = Dropout(drop_rate_hidden)(conv1)
    conv1 = Conv2D(nfilters, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(momentum=bn_momentum)(conv1)
    conv1 = Dropout(drop_rate_hidden)(conv1)

    # Rate 2
    conv3 = Conv2D(nfilters, (3, 3), dilation_rate=(2, 2), activation='relu',
                   padding='same', name="rate2_1")(conv1)
    conv3 = BatchNormalization(momentum=bn_momentum)(conv3)
    conv3 = Dropout(drop_rate_hidden)(conv3)
    conv3 = Conv2D(nfilters, (3, 3), dilation_rate=(2, 2), activation='relu',
                   padding='same', name="rate2_2")(conv3)
    conv3 = BatchNormalization(momentum=bn_momentum)(conv3)
    conv3 = Dropout(drop_rate_hidden)(conv3)

    # Branches for ASPP

    # Branch for 1x1
    conv3a = Conv2D(nfilters, (3, 3), activation='relu',
                    padding='same', name="branch1x1_1")(conv3)
    conv3a = BatchNormalization(momentum=bn_momentum)(conv3a)
    conv3a = Dropout(drop_rate_hidden)(conv3a)
    conv3a = Conv2D(nfilters, (1, 1), activation='relu',
                    padding='same', name="branch1x1_2")(conv3a)
    conv3a = BatchNormalization(momentum=bn_momentum)(conv3a)
    conv3a = Dropout(drop_rate_hidden)(conv3a)

    # Branch for 3x3 rate 6
    conv4 = Conv2D(nfilters, (3, 3), dilation_rate=(6, 6), activation='relu',
                   padding='same', name="rate6_1")(conv3)
    conv4 = BatchNormalization(momentum=bn_momentum)(conv4)
    conv4 = Dropout(drop_rate_hidden)(conv4)
    conv4 = Conv2D(nfilters, (3, 3), dilation_rate=(6, 6), activation='relu',
                   padding='same', name="rate6_2")(conv4)
    conv4 = BatchNormalization(momentum=bn_momentum)(conv4)
    conv4 = Dropout(drop_rate_hidden)(conv4)

    # Branch for 3x3 rate 12
    conv5 = Conv2D(nfilters, (3, 3), dilation_rate=(12, 12), activation='relu',
                   padding='same', name="rate12_1")(conv3)
    conv5 = BatchNormalization(momentum=bn_momentum)(conv5)
    conv5 = Dropout(drop_rate_hidden)(conv5)
    conv5 = Conv2D(nfilters, (3, 3), dilation_rate=(12, 12), activation='relu',
                   padding='same', name="rate12_2")(conv5)
    conv5 = BatchNormalization(momentum=bn_momentum)(conv5)
    conv5 = Dropout(drop_rate_hidden)(conv5)

    # Branch for 3x3 rate 18
    conv6 = Conv2D(nfilters, (3, 3), dilation_rate=(18, 18), activation='relu',
                   padding='same', name="rate18_1")(conv3)
    conv6 = BatchNormalization(momentum=bn_momentum)(conv6)
    conv6 = Dropout(drop_rate_hidden)(conv6)
    conv6 = Conv2D(nfilters, (3, 3), dilation_rate=(18, 18), activation='relu',
                   padding='same', name="rate18_2")(conv6)
    conv6 = BatchNormalization(momentum=bn_momentum)(conv6)
    conv6 = Dropout(drop_rate_hidden)(conv6)

    # Branch for 3x3 rate 24
    conv7 = Conv2D(nfilters, (3, 3), dilation_rate=(24, 24), activation='relu',
                   padding='same', name="rate24_1")(conv3)
    conv7 = BatchNormalization(momentum=bn_momentum)(conv7)
    conv7 = Dropout(drop_rate_hidden)(conv7)
    conv7 = Conv2D(nfilters, (3, 3), dilation_rate=(24, 24), activation='relu',
                   padding='same', name="rate24_2")(conv7)
    conv7 = BatchNormalization(momentum=bn_momentum)(conv7)
    conv7 = Dropout(drop_rate_hidden)(conv7)

    # Branch for the global context
    global_pool = GlobalAveragePooling2D()(conv1)
    global_pool = RepeatVector(CROP_HEIGHT * CROP_WIDTH)(global_pool)
    global_pool = Reshape((CROP_HEIGHT, CROP_WIDTH, nfilters))(global_pool)

    # Concatenation
    concat = concatenate([conv3a, conv4, conv5,
                          conv6, global_pool, conv7], axis=3)
    concat = BatchNormalization(momentum=bn_momentum)(concat)
    concat = Dropout(drop_rate_concat)(concat)

    amort = Conv2D(64, (1, 1), activation='relu',
                   padding='same', name="amort")(concat)
    amort = BatchNormalization(momentum=bn_momentum)(amort)
    amort = Dropout(drop_rate_hidden)(amort)

    predictions = Conv2D(1, (1, 1), activation='sigmoid',
                         padding='same', name="predictions")(amort)

    model = Model(inputs=[inputs], outputs=[predictions])

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt,
                  loss=dice_coef_loss,
                  metrics=["accuracy"])

    return model
