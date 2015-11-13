#!/usr/bin/env python
# ==========================================================================================
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
#
# License: see the LICENSE.TXT
# ==========================================================================================

import gzip
import sys
import os

import tensorflow.python.platform
import numpy
import tensorflow as tf

from msct_image import Image

TRAINING_SOURCE_DATA = '/Users/benjamindeleener/data/ismrm16_template/humanSpine_03_DTI/training/data/'
TRAINING_LABELS_DATA = '/Users/benjamindeleener/data/ismrm16_template/humanSpine_03_DTI/training/labels/'
TEST_SOURCE_DATA = '/Users/benjamindeleener/data/ismrm16_template/humanSpine_03_DTI/test/data/'
TEST_LABELS_DATA = '/Users/benjamindeleener/data/ismrm16_template/humanSpine_03_DTI/test/labels/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 53
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 50  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10

def extract_data(path_data):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    """
    print 'Extracting', path_data

    data = None
    for fname_im in os.listdir( path_data ):
        im_data = Image(path_data + fname_im)
        if data is None:
            data = numpy.expand_dims(im_data.data, axis=0)
        else:
            data = numpy.concatenate((data, numpy.expand_dims(im_data.data, axis=0)), axis=0)
    data = numpy.expand_dims(data, axis=3)
    print data.shape


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument

    train_data = extract_data(TRAINING_SOURCE_DATA)
    train_labels = extract_data(TRAINING_LABELS_DATA)
    test_data = extract_data(TEST_SOURCE_DATA)
    test_labels = extract_data(TEST_LABELS_DATA)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :, :, :]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, :, :, :]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    # For the validation and test data, we'll just hold the entire dataset in
    # one constant node.
    validation_data_node = tf.constant(validation_data)
    test_data_node = tf.constant(test_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    convA1_weights = tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS, 64],  # 3x3 filter, depth 64.
                                stddev=0.1, seed=SEED))
    convA1_biases = tf.Variable(tf.zeros([64]))
    convA2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64],  # 3x3 filter, depth 64.
                                stddev=0.1, seed=SEED))
    convA2_biases = tf.Variable(tf.zeros([64]))

    convB1_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128],  # 3x3 filter, depth 128.
                                stddev=0.1, seed=SEED))
    convB1_biases = tf.Variable(tf.zeros([128]))
    convB2_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 128],  # 3x3 filter, depth 128.
                                stddev=0.1, seed=SEED))
    convB2_biases = tf.Variable(tf.zeros([128]))

    convC1_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256],  # 3x3 filter, depth 256.
                                stddev=0.1, seed=SEED))
    convC1_biases = tf.Variable(tf.zeros([256]))
    convC2_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],  # 3x3 filter, depth 256.
                                stddev=0.1, seed=SEED))
    convC2_biases = tf.Variable(tf.zeros([256]))

    convD1_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 512],  # 3x3 filter, depth 512.
                                stddev=0.1, seed=SEED))
    convD1_biases = tf.Variable(tf.zeros([512]))
    convD2_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
                                stddev=0.1, seed=SEED))
    convD2_biases = tf.Variable(tf.zeros([512]))

    convE1_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 1024],  # 3x3 filter, depth 1024.
                                stddev=0.1, seed=SEED))
    convE1_biases = tf.Variable(tf.zeros([1024]))
    convE2_weights = tf.Variable(tf.truncated_normal([3, 3, 1024, 1024],  # 3x3 filter, depth 1024.
                                stddev=0.1, seed=SEED))
    convE2_biases = tf.Variable(tf.zeros([1024]))

    upconvEF_weights = tf.Variable(tf.truncated_normal([2, 2, 1024, 512],  # 2x2 filter, depth 512.
                                stddev=0.1, seed=SEED))

    convF1_weights = tf.Variable(tf.truncated_normal([3, 3, 1024, 512],  # 3x3 filter, depth 512.
                                stddev=0.1, seed=SEED))
    convF1_biases = tf.Variable(tf.zeros([512]))
    convF2_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 512],  # 3x3 filter, depth 512.
                                stddev=0.1, seed=SEED))
    convF2_biases = tf.Variable(tf.zeros([512]))

    upconvFG_weights = tf.Variable(tf.truncated_normal([2, 2, 512, 256],  # 2x2 filter, depth 256.
                                stddev=0.1, seed=SEED))

    convG1_weights = tf.Variable(tf.truncated_normal([3, 3, 512, 256],  # 3x3 filter, depth 256.
                                stddev=0.1, seed=SEED))
    convG1_biases = tf.Variable(tf.zeros([256]))
    convG2_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],  # 3x3 filter, depth 256.
                                stddev=0.1, seed=SEED))
    convG2_biases = tf.Variable(tf.zeros([256]))

    upconvGH_weights = tf.Variable(tf.truncated_normal([2, 2, 256, 128],  # 2x2 filter, depth 128.
                                stddev=0.1, seed=SEED))

    convH1_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 128],  # 3x3 filter, depth 128.
                                stddev=0.1, seed=SEED))
    convH1_biases = tf.Variable(tf.zeros([128]))
    convH2_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 128],  # 3x3 filter, depth 128.
                                stddev=0.1, seed=SEED))
    convH2_biases = tf.Variable(tf.zeros([128]))

    upconvHI_weights = tf.Variable(tf.truncated_normal([2, 2, 128, 64],  # 2x2 filter, depth 64.
                                stddev=0.1, seed=SEED))

    convI1_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 64],  # 3x3 filter, depth 64.
                                stddev=0.1, seed=SEED))
    convI1_biases = tf.Variable(tf.zeros([64]))
    convI2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 64],  # 3x3 filter, depth 64.
                                stddev=0.1, seed=SEED))
    convI2_biases = tf.Variable(tf.zeros([64]))

    finalconv_weights = tf.Variable(tf.truncated_normal([1, 1, 64, 2],  # 2x2 filter, depth 64.
                                stddev=0.1, seed=SEED))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2X 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        # Bias and rectified linear non-linearity.
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.

        # contraction A
        convA1 = tf.nn.conv2d(data, convA1_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluA1 = tf.nn.relu(tf.nn.bias_add(convA1, convA1_biases))
        convA2 = tf.nn.conv2d(reluA1, convA2_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluA2 = tf.nn.relu(tf.nn.bias_add(convA2, convA2_biases))

        pool = tf.nn.max_pool(reluA2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # contraction B
        convB1 = tf.nn.conv2d(pool, convB1_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluB1 = tf.nn.relu(tf.nn.bias_add(convB1, convB1_biases))
        convB2 = tf.nn.conv2d(reluB1, convB2_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluB2 = tf.nn.relu(tf.nn.bias_add(convB2, convB2_biases))

        pool = tf.nn.max_pool(reluB2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # contraction C
        convC1 = tf.nn.conv2d(pool, convC1_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluC1 = tf.nn.relu(tf.nn.bias_add(convC1, convC1_biases))
        convC2 = tf.nn.conv2d(reluC1, convC2_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluC2 = tf.nn.relu(tf.nn.bias_add(convC2, convC2_biases))

        pool = tf.nn.max_pool(reluC2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # contraction D
        convD1 = tf.nn.conv2d(pool, convD1_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluD1 = tf.nn.relu(tf.nn.bias_add(convD1, convD1_biases))
        convD2 = tf.nn.conv2d(reluD1, convD2_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluD2 = tf.nn.relu(tf.nn.bias_add(convD2, convD2_biases))

        pool = tf.nn.max_pool(reluD2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # contraction E
        convE1 = tf.nn.conv2d(pool, convE1_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluE1 = tf.nn.relu(tf.nn.bias_add(convE1, convE1_biases))
        convE2 = tf.nn.conv2d(reluE1, convE2_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluE2 = tf.nn.relu(tf.nn.bias_add(convE2, convE2_biases))

        # up-convolution:
        # 2x2 convolution with upsampling by a factor 2, then concatenation
        resample = reluE2.repeat(2, axis=1).repeat(2, axis=2)
        upconv = tf.nn.conv2d(resample, upconvEF_weights, strides=[1, 1, 1, 1], padding='SAME')
        b_min = (reluD2.shape[1] - upconv.shape[1]) / 2 - 1
        b_max = b_min + upconv.shape[1] + 1
        upconv_concat = tf.concat(concat_dim=0, values=[reluD2[:, b_min:b_max, b_min:b_max, :], upconv])

        # expansion F
        convF1 = tf.nn.conv2d(upconv_concat, convF1_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluF1 = tf.nn.relu(tf.nn.bias_add(convF1, convF1_biases))
        convF2 = tf.nn.conv2d(reluF1, convF2_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluF2 = tf.nn.relu(tf.nn.bias_add(convF2, convF2_biases))

        # up-convolution
        resample = reluF2.repeat(2, axis=1).repeat(2, axis=2)
        upconv = tf.nn.conv2d(resample, upconvFG_weights, strides=[1, 1, 1, 1], padding='SAME')
        b_min = (reluC2.shape[1] - upconv.shape[1]) / 2 - 1
        b_max = b_min + upconv.shape[1] + 1
        upconv_concat = tf.concat(concat_dim=0, values=[reluC2[:, b_min:b_max, b_min:b_max, :], upconv])

        # expansion G
        convG1 = tf.nn.conv2d(upconv_concat, convG1_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluG1 = tf.nn.relu(tf.nn.bias_add(convG1, convG1_biases))
        convG2 = tf.nn.conv2d(reluG1, convG2_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluG2 = tf.nn.relu(tf.nn.bias_add(convG2, convG2_biases))

        # up-convolution
        resample = reluG2.repeat(2, axis=1).repeat(2, axis=2)
        upconv = tf.nn.conv2d(resample, upconvGH_weights, strides=[1, 1, 1, 1], padding='SAME')
        b_min = (reluB2.shape[1] - upconv.shape[1]) / 2 - 1
        b_max = b_min + upconv.shape[1] + 1
        upconv_concat = tf.concat(concat_dim=0, values=[reluB2[:, b_min:b_max, b_min:b_max, :], upconv])

        # expansion H
        convH1 = tf.nn.conv2d(upconv_concat, convH1_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluH1 = tf.nn.relu(tf.nn.bias_add(convH1, convH1_biases))
        convH2 = tf.nn.conv2d(reluH1, convH2_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluH2 = tf.nn.relu(tf.nn.bias_add(convH2, convH2_biases))

        # up-convolution
        resample = reluH2.repeat(2, axis=1).repeat(2, axis=2)
        upconv = tf.nn.conv2d(resample, upconvHI_weights, strides=[1, 1, 1, 1], padding='SAME')
        b_min = (reluA2.shape[1]-upconv.shape[1])/2-1
        b_max = b_min + upconv.shape[1] + 1
        upconv_concat = tf.concat(concat_dim=0, values=[reluA2[:, b_min:b_max, b_min:b_max, :], upconv])

        # expansion I
        convI1 = tf.nn.conv2d(upconv_concat, convI1_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluI1 = tf.nn.relu(tf.nn.bias_add(convI1, convI1_biases))
        convI2 = tf.nn.conv2d(reluI1, convI2_weights, strides=[1, 1, 1, 1], padding='SAME')
        reluI2 = tf.nn.relu(tf.nn.bias_add(convI2, convI2_biases))

        finalconv = tf.nn.conv2d(reluI2, finalconv_weights, strides=[1, 1, 1, 1], padding='SAME')

        return finalconv


    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    validation_prediction = tf.nn.softmax(model(validation_data_node))
    test_prediction = tf.nn.softmax(model(test_data_node))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print 'Initialized!'
        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            if step % 100 == 0:
                print 'Epoch %.2f' % (float(step) * BATCH_SIZE / train_size)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch error: %.1f%%' % error_rate(predictions,
                                                             batch_labels)
                print 'Validation error: %.1f%%' % error_rate(
                    validation_prediction.eval(), validation_labels)
                sys.stdout.flush()
        # Finally print the result!
        test_error = error_rate(test_prediction.eval(), test_labels)
        print 'Test error: %.1f%%' % test_error


if __name__ == '__main__':
    tf.app.run()