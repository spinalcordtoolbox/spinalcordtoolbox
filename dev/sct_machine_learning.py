#!/usr/bin/env python
# ==========================================================================================
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
#
# License: see the LICENSE.TXT
# ==========================================================================================

import sys
import os

import tensorflow.python.platform
import numpy
import tensorflow as tf

import sct_utils as sct
from msct_image import Image

TRAINING_SOURCE_DATA = '/Users/benjamindeleener/data/ismrm16_template/humanSpine_03_DTI/training/data/'
TRAINING_LABELS_DATA = '/Users/benjamindeleener/data/ismrm16_template/humanSpine_03_DTI/training/labels/'
TEST_SOURCE_DATA = '/Users/benjamindeleener/data/ismrm16_template/humanSpine_03_DTI/test/data/'
TEST_LABELS_DATA = '/Users/benjamindeleener/data/ismrm16_template/humanSpine_03_DTI/test/labels/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 80
NUM_CHANNELS = 1
NUM_LABELS = 2
VALIDATION_SIZE = 50  # Size of the validation set.
SEED = 66478  # Set to None for random seed. or 66478
BATCH_SIZE = 35
NUM_EPOCHS = 100


class UNetModel:
    def __init__(self, image_size, depth=4):
        self.depth = depth
        self.num_features = 64
        num_features = self.num_features
        num_features_init = NUM_CHANNELS
        self.num_classes = 2
        self.image_size = image_size

        self.weights_contraction = []
        self.weights_expansion = []
        self.upconv_weights = []
        # Setting variables that will be optimized
        # contraction
        for i in range(self.depth):
            self.weights_contraction.append({'conv1': tf.Variable(tf.truncated_normal([3, 3, num_features_init, num_features], stddev=0.1, seed=SEED)),
                                             'bias1': tf.Variable(tf.zeros([num_features])),
                                             'conv2': tf.Variable(tf.truncated_normal([3, 3, num_features, num_features], stddev=0.1, seed=SEED)),
                                             'bias2': tf.Variable(tf.zeros([num_features]))})
            num_features_init = num_features
            num_features = num_features_init * 2

        self.weights_bottom_layer = {
            'conv1': tf.Variable(tf.truncated_normal([3, 3, num_features_init, num_features], stddev=0.1, seed=SEED)),
            'bias1': tf.Variable(tf.zeros([num_features])),
            'conv2': tf.Variable(tf.truncated_normal([3, 3, num_features, num_features], stddev=0.1, seed=SEED)),
            'bias2': tf.Variable(tf.zeros([num_features]))}

        # expansion
        num_features_init = num_features
        num_features = num_features_init / 2
        for i in range(depth):
            self.upconv_weights.append(tf.Variable(tf.truncated_normal([2, 2, num_features_init, num_features], stddev=0.1, seed=SEED)))
            self.weights_expansion.append({'conv1': tf.Variable(tf.truncated_normal([3, 3, num_features_init, num_features], stddev=0.1, seed=SEED)),
                                           'bias1': tf.Variable(tf.zeros([num_features])),
                                           'conv2': tf.Variable(tf.truncated_normal([3, 3, num_features, num_features], stddev=0.1, seed=SEED)),
                                           'bias2': tf.Variable(tf.zeros([num_features]))})
            num_features_init = num_features
            num_features = num_features_init / 2

        self.finalconv_weights = tf.Variable(tf.truncated_normal([1, 1, num_features * 2, self.num_classes], stddev=0.1, seed=SEED))

    def model(self, data, train=False):
        """The Model definition.
        # 2X 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        # Bias and rectified linear non-linearity.
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        """
        # contraction
        image_size_temp = [self.image_size]
        data_temp = data
        relu_results = []
        for i in range(self.depth):
            conv = tf.nn.conv2d(data_temp, self.weights_contraction[i]['conv1'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_contraction[i]['bias1']))
            conv = tf.nn.conv2d(relu, self.weights_contraction[i]['conv2'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_contraction[i]['bias2']))
            relu_results.append(relu)

            data_temp = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            image_size_temp.append((image_size_temp[-1]) / 2)

        # convolution of bottom layer
        conv = tf.nn.conv2d(data_temp, self.weights_bottom_layer['conv1'], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_bottom_layer['bias1']))
        conv = tf.nn.conv2d(relu, self.weights_bottom_layer['conv2'], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_bottom_layer['bias2']))
        image_size_temp.append(image_size_temp[-1])

        # expansion
        for i in range(self.depth):
            # up-convolution:
            # 2x2 convolution with upsampling by a factor 2, then concatenation
            resample = tf.image.resize_images(relu, image_size_temp[-1] * 2, image_size_temp[-1] * 2)
            upconv = tf.nn.conv2d(resample, self.upconv_weights[i], strides=[1, 1, 1, 1], padding='SAME')
            image_size_temp.append(image_size_temp[-1] * 2)
            upconv_concat = tf.concat(concat_dim=3, values=[tf.slice(relu_results[self.depth-i-1], [0, 0, 0, 0], [-1, image_size_temp[self.depth-i] * 2, image_size_temp[self.depth-i] * 2, -1]), upconv])

            # expansion
            conv = tf.nn.conv2d(upconv_concat, self.weights_expansion[i]['conv1'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_expansion[i]['bias1']))
            conv = tf.nn.conv2d(relu, self.weights_expansion[i]['conv2'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_expansion[i]['bias2']))

        finalconv = tf.nn.conv2d(relu, self.finalconv_weights, strides=[1, 1, 1, 1], padding='SAME')
        final_result = tf.reshape(finalconv, tf.TensorShape([finalconv.get_shape().as_list()[0] * image_size_temp[-1] * image_size_temp[-1], NUM_LABELS]))

        return final_result


def extract_data(path_data, offset_size=0):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    """
    ignore_list = ['.DS_Store']
    print 'Extracting', path_data

    data = None
    for fname_im in os.listdir(path_data):
        if fname_im in ignore_list:
            continue
        im_data = Image(path_data + fname_im)
        if offset_size == 0:
            data_image = im_data.data
        else:
            data_image = im_data.data[offset_size:-offset_size, offset_size:-offset_size]
            data_image = (data_image - numpy.min(data_image)) / (numpy.max(data_image) - numpy.min(data_image))
        if data is None:
            data = numpy.expand_dims(data_image, axis=0)
        else:
            data = numpy.concatenate((data, numpy.expand_dims(data_image, axis=0)), axis=0)
    data = numpy.expand_dims(data, axis=3)
    print data.shape
    return data.astype(numpy.float32)


def extract_label(path_data, offset_size=0):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    """
    ignore_list = ['.DS_Store']
    print 'Extracting', path_data

    data = None
    for fname_im in os.listdir(path_data):
        if fname_im in ignore_list:
            continue
        im_data = Image(path_data + fname_im)
        if offset_size == 0:
            data_image = im_data.data
        else:
            data_image = im_data.data[offset_size:-offset_size, offset_size:-offset_size]
        if data is None:
            data = numpy.expand_dims(data_image, axis=0)
        else:
            data = numpy.concatenate((data, numpy.expand_dims(data_image, axis=0)), axis=0)
    data = numpy.expand_dims(data, axis=3)
    data = numpy.concatenate((1-data, data), axis=3)
    print data.shape
    return data.astype(numpy.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    # Dice coefficients between two numpy arrays
    im1 = numpy.asarray(predictions).astype(numpy.bool)
    im2 = numpy.asarray(labels).astype(numpy.bool)
    intersection = numpy.logical_and(im1, im2)
    return 100. - 100. * 2. * intersection.sum() / (im1.sum() + im2.sum())
    # return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) / predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument

    # Setting U-net parameters
    depth = 3

    # Make sure image size corresponds to requirements
    # "select the input tile size such that all 2x2 max-pooling operationsare applied to a
    # layer with an even x- and y-size."
    image_size_temp = IMAGE_SIZE
    for i in range(depth):
        if image_size_temp % 2 != 0:
            sct.printv('ERROR: image size must satisfy requirements (select the input tile size such that all 2x2 '
                       'max-pooling operationsare applied to a layer with an even x- and y-size.)', type='error')
        image_size_temp = (image_size_temp) / 2
    image_size_bottom = image_size_temp

    # Compute the size of the image segmentation, based on depth
    for i in range(depth):
        image_size_temp *= 2
    segmentation_image_size = image_size_temp
    offset_images = (IMAGE_SIZE - segmentation_image_size) / 2

    sct.printv('Original image size = ' + str(IMAGE_SIZE))
    sct.printv('Image size at bottom layer = ' + str(image_size_bottom))
    sct.printv('Image size of output = ' + str(segmentation_image_size))

    # Extracting datasets
    train_data = extract_data(TRAINING_SOURCE_DATA)
    train_labels = extract_label(TRAINING_LABELS_DATA, offset_images)
    test_data = extract_data(TEST_SOURCE_DATA)
    test_labels = extract_label(TEST_LABELS_DATA, offset_images)
    test_labels = numpy.reshape(test_labels, [test_labels.shape[0] * test_labels.shape[1] * test_labels.shape[2], NUM_LABELS])


    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :, :, :]
    validation_labels = train_labels[:VALIDATION_SIZE]
    validation_labels = numpy.reshape(validation_labels, [validation_labels.shape[0] * validation_labels.shape[1] * validation_labels.shape[2], NUM_LABELS])
    train_data = train_data[VALIDATION_SIZE:, :, :, :]
    train_labels = train_labels[VALIDATION_SIZE:]
    #train_labels = numpy.reshape(train_labels, [train_labels.shape[0] * train_labels.shape[1] * train_labels.shape[2], train_labels.shape[3]])
    num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE * segmentation_image_size * segmentation_image_size, NUM_LABELS))
    # For the validation and test data, we'll just hold the entire dataset in
    # one constant node.
    validation_data_node = tf.constant(validation_data)
    test_data_node = tf.constant(test_data)

    unet = UNetModel(IMAGE_SIZE, depth)

    # Training computation: logits + cross-entropy loss.
    logits = unet.model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node))

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(0.001,  # Base learning rate.
                                               batch * BATCH_SIZE,  # Current index into the dataset.
                                               train_size,  # Decay step.
                                               0.95,  # Decay rate.
                                               staircase=True)
    # Use simple gradient descent for the optimization.
    # learning_rate = 0.001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)
    #optimizer = tf.train.MomentumOptimizer(learning_rate).minimize(loss, global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    validation_prediction = tf.nn.softmax(unet.model(validation_data_node))
    test_prediction = tf.nn.softmax(unet.model(test_data_node))

    # Create a local session to run this computation.
    import multiprocessing as mp
    number_of_cores = mp.cpu_count()
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=number_of_cores, intra_op_parallelism_threads=number_of_cores)) as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print 'Initialized!'
        # Loop through training steps.
        number_of_step = int(num_epochs * train_size / BATCH_SIZE)
        print 'Number of step = ' + str(number_of_step)
        timer_training = sct.Timer(number_of_step)
        timer_training.start()
        for step in xrange(int(num_epochs * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            batch_labels = numpy.reshape(batch_labels, [batch_labels.shape[0] * batch_labels.shape[1] * batch_labels.shape[2], NUM_LABELS])
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
            if step % 100 == 0:
                print 'Epoch ' + str(round(float(step) * BATCH_SIZE / train_size, 2)) + ' %'
                timer_training.iterations_done(step)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch error: %.1f%%' % error_rate(predictions, batch_labels)
                print 'Validation error: %.1f%%' % error_rate(validation_prediction.eval(), validation_labels)
            elif step % 10 == 0:
                print 'Epoch ' + str(round(float(step) * BATCH_SIZE / train_size, 2)) + ' %'
                timer_training.iterations_done(step)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
        # Finally print the result!
        test_error = error_rate(test_prediction.eval(), test_labels)
        print 'Test error: %.1f%%' % test_error
        timer_training.printTotalTime()

        import pickle
        pickle.dump(unet, open("unet-model.p", "wb"))


if __name__ == '__main__':
    tf.app.run()
