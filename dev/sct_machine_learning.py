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
IMAGE_SIZE = 60
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

    depth = 4
    num_features = 64
    num_features_init = NUM_CHANNELS
    num_classes = 2
    weights_contraction = []
    weights_expansion = []
    upconv_weights = []

    # contraction
    for i in range(depth):
        weights_contraction[i] = {'conv1': tf.Variable(tf.truncated_normal([3, 3, num_features_init, num_features], stddev=0.1, seed=SEED)),
                                  'bias1': tf.Variable(tf.zeros([num_features])),
                                  'conv2': tf.Variable(tf.truncated_normal([3, 3, num_features, num_features], stddev=0.1, seed=SEED)),
                                  'bias2': tf.Variable(tf.zeros([num_features]))}
        num_features_init = num_features
        num_features = num_features_init * 2

    weights_bottom_layer = {'conv1': tf.Variable(tf.truncated_normal([3, 3, num_features_init, num_features], stddev=0.1, seed=SEED)),
                            'bias1': tf.Variable(tf.zeros([num_features])),
                            'conv2': tf.Variable(tf.truncated_normal([3, 3, num_features, num_features], stddev=0.1, seed=SEED)),
                            'bias2': tf.Variable(tf.zeros([num_features]))}

    # expansion
    num_features_init = num_features
    num_features = num_features_init / 2
    for i in range(depth):
        upconv_weights[i] = tf.Variable(tf.truncated_normal([2, 2, num_features_init, num_features], stddev=0.1, seed=SEED))
        weights_expansion[i] = {'conv1': tf.Variable(tf.truncated_normal([3, 3, num_features_init, num_features], stddev=0.1, seed=SEED)),
                                'bias1': tf.Variable(tf.zeros([num_features])),
                                'conv2': tf.Variable(tf.truncated_normal([3, 3, num_features, num_features], stddev=0.1, seed=SEED)),
                                'bias2': tf.Variable(tf.zeros([num_features]))}
        num_features_init = num_features
        num_features = num_features_init / 2

    finalconv_weights = tf.Variable(tf.truncated_normal([1, 1, num_features, num_classes], stddev=0.1, seed=SEED))

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

        # contraction
        data_temp = data
        relu_results = []
        for i in range(depth):
            conv = tf.nn.conv2d(data_temp, weights_contraction[i]['conv1'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, weights_contraction[i]['bias1']))
            conv = tf.nn.conv2d(relu, weights_contraction[i]['conv2'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, weights_contraction[i]['bias2']))
            relu_results.append(relu)

            data_temp = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # convolution of bottom layer
        conv = tf.nn.conv2d(data_temp, weights_bottom_layer['conv1'], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, weights_bottom_layer['bias1']))
        conv = tf.nn.conv2d(relu, weights_bottom_layer['conv2'], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, weights_bottom_layer['bias2']))

        # expansion
        for i in range(depth):
            # up-convolution:
            # 2x2 convolution with upsampling by a factor 2, then concatenation
            resample = relu.repeat(2, axis=1).repeat(2, axis=2)
            upconv = tf.nn.conv2d(resample, upconv_weights[i], strides=[1, 1, 1, 1], padding='SAME')
            b_min = (relu_results[depth-i-1].shape[1] - upconv.shape[1]) / 2 - 1
            b_max = b_min + upconv.shape[1] + 1
            upconv_concat = tf.concat(concat_dim=0, values=[relu_results[depth-i-1][:, b_min:b_max, b_min:b_max, :], upconv])

            # expansion F
            conv = tf.nn.conv2d(upconv_concat, weights_expansion[i]['conv1'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, weights_expansion[i]['bias1']))
            conv = tf.nn.conv2d(relu, weights_expansion[i]['conv2'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, weights_expansion[i]['bias2']))

        finalconv = tf.nn.conv2d(relu, finalconv_weights, strides=[1, 1, 1, 1], padding='SAME')

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