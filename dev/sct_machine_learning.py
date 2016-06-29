#!/usr/bin/env python
# ==========================================================================================
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
#
# License: see the LICENSE.TXT
# ==========================================================================================

from __future__ import absolute_import

import sys
import os

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy
import tensorflow as tf
from random import shuffle

import sct_utils as sct
from msct_image import Image
import math

try:
   import cPickle as pickle
except:
   import pickle

path_data = '/home/neuropoly/data/spinal_cord_segmentation_data/'
output_path = ''
TRAINING_SOURCE_DATA = path_data+'training/data/'
TRAINING_LABELS_DATA = path_data+'training/labels/'
TEST_SOURCE_DATA = path_data+'test/data/'
TEST_LABELS_DATA = path_data+'test/labels/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 80
NUM_CHANNELS = 1
NUM_LABELS = 2
VALIDATION_SIZE = 256  # Size of the validation set.
SEED = None  # Set to None for random seed. or 66478
BATCH_SIZE = 256
NUM_EPOCHS = 100

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/unet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")


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
            self.weights_contraction.append({'conv1': tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init))), seed=SEED)),
                                             'bias1': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))),
                                             'conv2': tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))), seed=SEED)),
                                             'bias2': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))))})
            num_features_init = num_features
            num_features = num_features_init * 2

        self.weights_bottom_layer = {
            'conv1': tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init))), seed=SEED)),
            'bias1': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))),
            'conv2': tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))), seed=SEED)),
            'bias2': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))))}

        # expansion
        num_features_init = num_features
        num_features = num_features_init / 2
        for i in range(depth):
            self.upconv_weights.append(tf.Variable(tf.random_normal([2, 2, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init))), seed=SEED)))
            self.weights_expansion.append({'conv1': tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init))), seed=SEED)),
                                           'bias1': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))),
                                           'conv2': tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))), seed=SEED)),
                                           'bias2': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))))})
            num_features_init = num_features
            num_features = num_features_init / 2

        self.finalconv_weights = tf.Variable(tf.random_normal([1, 1, num_features * 2, self.num_classes], stddev=math.sqrt(2.0/(9.0*float(num_features*2))), seed=SEED))

    def save_parameters(self, fname_out=''):
        pickle.dump(self.weights_contraction, open("unet-model-weights_contraction.p", "wb"))
        pickle.dump(self.weights_bottom_layer, open("unet-model-weights_bottom_layer.p", "wb"))
        pickle.dump(self.upconv_weights, open("unet-model-upconv_weights.p", "wb"))
        pickle.dump(self.weights_expansion, open("unet-model-weights_expansion.p", "wb"))
        pickle.dump(self.finalconv_weights, open("unet-model-finalconv_weights.p", "wb"))
        if not fname_out:
            fname_out = 'unet-model.gz'
        sct.run('gzip unet-model-*  > ' + fname_out)

    def load_parameters(self, fname_in):
        sct.printv('out')

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


def extract_data(path_data, offset_size=0, list_images=None, verbose=1):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    """
    from sys import stdout
    ignore_list = ['.DS_Store']
    if verbose == 1:
        sct.printv('Extracting '+ path_data)
    cr = '\r'

    data = []
    list_data = []
    images_folder = os.listdir(path_data)
    if list_images is None:
        for i, fname_im in enumerate(images_folder):
            if verbose == 1:
                stdout.write(cr)
                stdout.write(str(i) + '/' + str(len(images_folder)))
            if fname_im in ignore_list:
                continue
            list_data.append(fname_im)

        if verbose == 1:
            stdout.write(cr)
            sct.printv('Done.        ')
        return list_data
    else:
        for i, fname_im in enumerate(list_images):
            if verbose == 1:
                stdout.write(cr)
                stdout.write(str(i) + '/' + str(len(list_images)))
            im_data = Image(path_data + fname_im)
            if offset_size == 0:
                data_image = im_data.data
            else:
                data_image = im_data.data[offset_size:-offset_size, offset_size:-offset_size]
                data_image = (data_image - numpy.min(data_image)) / (numpy.max(data_image) - numpy.min(data_image))
            data.append(numpy.expand_dims(data_image, axis=0))

        data_stack = numpy.concatenate(data, axis=0)
        data = numpy.expand_dims(data_stack, axis=3)
        if verbose == 1:
            stdout.write(cr)
            sct.printv(data.shape)
        return data.astype(numpy.float32)


def extract_label(path_data, segmentation_image_size=0, list_images=None, verbose=1):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    """
    from sys import stdout
    cr = '\r'
    offset_size = (IMAGE_SIZE - segmentation_image_size) / 2
    number_pixel = segmentation_image_size * segmentation_image_size
    ignore_list = ['.DS_Store']
    if verbose == 1:
        sct.printv('Extracting' + path_data)

    data, weights = [], []
    list_data = []
    images_folder = os.listdir(path_data)
    if list_images is None:
        for i, fname_im in enumerate(images_folder):
            if verbose == 1:
                stdout.write(cr)
                stdout.write(str(i) + '/' + str(len(images_folder)))
            if fname_im in ignore_list:
                continue
            list_data.append(fname_im)

        if verbose == 1:
            stdout.write(cr)
            sct.printv('Done.        ')
        return list_data
    else:
        for i, fname_im in enumerate(list_images):
            if verbose == 1:
                stdout.write(cr)
                stdout.write(str(i) + '/' + str(len(list_images)))
            im_data = Image(path_data + fname_im)
            if offset_size == 0:
                data_image = im_data.data
            else:
                data_image = im_data.data[offset_size:-offset_size, offset_size:-offset_size]
            number_of_segpixels = numpy.count_nonzero(data_image)
            weights_image = data_image * number_of_segpixels / number_pixel + (1 - data_image) * (number_pixel - number_of_segpixels) / number_pixel
            data.append(numpy.expand_dims(data_image, axis=0))
            weights.append(numpy.expand_dims(weights_image, axis=0))

        data_stack = numpy.concatenate(data, axis=0)
        weights_stack = numpy.concatenate(weights, axis=0)
        data = numpy.expand_dims(data_stack, axis=3)
        data = numpy.concatenate((1-data, data), axis=3)
        if verbose == 1:
            stdout.write(cr)
            sct.printv(data.shape)
        return data.astype(numpy.float32), weights_stack.astype(numpy.float32)


def savePredictions(predictions, path_output, list_images, segmentation_image_size):
    number_of_images = len(list_images)
    predictions = numpy.reshape(predictions, [number_of_images, segmentation_image_size, segmentation_image_size, NUM_LABELS])
    predictions = predictions[:, :, :, 1]
    for i, pref in enumerate(predictions):
        im_pred = Image(pref)
        im_pred.setFileName(path_output+sct.add_suffix(list_images[i], '_pred'))
        im_pred.save()


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    # Dice coefficients between two numpy arrays
    predictions = predictions[:, 1]
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    im1 = numpy.asarray(predictions).astype(numpy.bool)
    labels = labels[:, 1]
    labels[labels >= 0.5] = 1
    labels[labels < 0.5] = 0
    im2 = numpy.asarray(labels).astype(numpy.bool)
    intersection = numpy.logical_and(im1, im2)

    return 100. - 100. * 2. * intersection.sum() / (im1.sum() + im2.sum())


def main(argv=None):  # pylint: disable=unused-argument

    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)

    with tf.Graph().as_default():
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
        # offset_images = (IMAGE_SIZE - segmentation_image_size) / 2

        sct.printv('Original image size = ' + str(IMAGE_SIZE))
        sct.printv('Image size at bottom layer = ' + str(image_size_bottom))
        sct.printv('Image size of output = ' + str(segmentation_image_size))

        # Extracting datasets
        list_data = extract_data(TRAINING_SOURCE_DATA)
        list_labels = extract_label(TRAINING_LABELS_DATA)
        list_test_data = extract_data(TEST_SOURCE_DATA)
        list_test_labels = extract_label(TEST_LABELS_DATA)

        # Generate a validation set
        validation_data = list_data[:VALIDATION_SIZE]
        validation_labels = list_labels[:VALIDATION_SIZE]
        list_data = list_data[VALIDATION_SIZE:]
        list_labels = list_labels[VALIDATION_SIZE:]
        num_epochs = NUM_EPOCHS
        train_size = len(list_labels)

        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE * segmentation_image_size * segmentation_image_size, NUM_LABELS))
        train_labels_weights = tf.placeholder(tf.float32, shape=(BATCH_SIZE * segmentation_image_size * segmentation_image_size))
        # For the validation and test data, we'll just hold the entire dataset in one constant node.

        unet = UNetModel(IMAGE_SIZE, depth)

        # Training computation: logits + cross-entropy loss.
        logits = unet.model(train_data_node, True)
        loss = tf.reduce_mean(tf.mul(train_labels_weights, tf.nn.softmax_cross_entropy_with_logits(logits, train_labels_node)))
        tf.scalar_summary('Loss', loss)

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0, trainable=False)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(0.001,  # Base learning rate.
                                                   batch * BATCH_SIZE,  # Current index into the dataset.
                                                   train_size,  # Decay step.
                                                   0.95,  # Decay rate.
                                                   staircase=True)
        tf.scalar_summary('Learning rate', learning_rate)

        error_rate_batch = tf.Variable(0.0, name='batch_error_rate', trainable=False)
        tf.scalar_summary('Batch error rate', error_rate_batch)

        error_rate_validation = tf.Variable(0.0, name='validation_error_rate', trainable=False)
        tf.scalar_summary('Validation error rate', error_rate_validation)

        # Use simple gradient descent for the optimization.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
        #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)
        #optimizer = tf.train.MomentumOptimizer(learning_rate).minimize(loss, global_step=batch)

        # Predictions for the minibatch, validation set and test set.
        train_prediction = tf.nn.softmax(logits)

        # Create a local session to run this computation.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        import multiprocessing as mp
        import time
        number_of_cores = mp.cpu_count()
        with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, inter_op_parallelism_threads=number_of_cores, intra_op_parallelism_threads=number_of_cores)) as s:
            # Run all the initializers to prepare the trainable parameters.
            init = tf.initialize_all_variables()
            s.run(init)
            tf.train.start_queue_runners(sess=s)
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=s.graph_def)

            sct.printv('\nShuffling batches!')
            number_of_step = int(num_epochs * train_size / BATCH_SIZE)
            steps = range(number_of_step)
            shuffle(steps)
            sct.printv('Initialized!')
            sct.printv('Number of step = ' + str(number_of_step))
            timer_training = sct.Timer(number_of_step)
            timer_training.start()
            # Loop through training steps.
            for i, step in enumerate(steps):
                sct.printv('Step '+ str(i) + '/' + str(len(steps)))
                sct.printv('Epoch ' + str(round(float(i) * BATCH_SIZE / train_size, 2)) + ' %')
                timer_training.iterations_done(i)
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                batch_data = extract_data(TRAINING_SOURCE_DATA, list_images=list_data[offset:(offset + BATCH_SIZE)], verbose=0)
                batch_labels, batch_labels_weights = extract_label(TRAINING_LABELS_DATA, segmentation_image_size, list_labels[offset:(offset + BATCH_SIZE)], verbose=0)
                batch_labels = numpy.reshape(batch_labels, [batch_labels.shape[0] * batch_labels.shape[1] * batch_labels.shape[2], NUM_LABELS])
                batch_labels_weights = numpy.reshape(batch_labels_weights, [batch_labels_weights.shape[0] * batch_labels_weights.shape[1] * batch_labels_weights.shape[2]])
                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph is should be fed to.
                feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels, train_labels_weights: batch_labels_weights}
                # Run the graph and fetch some of the nodes.
                _, l, lr, predictions = s.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)

                assert not numpy.isnan(l), 'Model diverged with loss = NaN'

                """
                if i % 100 == 0 or (i + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(s, checkpoint_path, global_step=i)
                """

                if i != 0 and i % 50 == 0:
                    error_rate_batch_tens = error_rate_batch.assign(error_rate(predictions, batch_labels))
                    validation_data_b = extract_data(TRAINING_SOURCE_DATA, list_images=validation_data, verbose=0)
                    validation_labels_b, validation_labels_weights = extract_label(TRAINING_LABELS_DATA, segmentation_image_size, validation_labels, verbose=0)
                    validation_labels_b = numpy.reshape(validation_labels_b, [validation_labels_b.shape[0] * validation_labels_b.shape[1] * validation_labels_b.shape[2], NUM_LABELS])
                    validation_data_node = tf.constant(validation_data_b)
                    validation_prediction = tf.nn.softmax(unet.model(validation_data_node))
                    error_rate_validation_tens = error_rate_validation.assign(error_rate(validation_prediction.eval(), validation_labels_b))
                else:
                    error_rate_batch_tens = error_rate_validation.assign(error_rate_batch.eval())
                    error_rate_validation_tens = error_rate_validation.assign(error_rate_validation.eval())

                if i != 0 and i % 5 == 0:
                    result = s.run([summary_op, learning_rate, error_rate_batch_tens, error_rate_validation_tens], feed_dict=feed_dict)
                    summary_str = result[0]
                    sct.printv('Minibatch loss: %.6f, learning rate: %.6f, error batch %.3f, error validation %.3f' % (l, lr, error_rate_batch.eval(), error_rate_validation.eval()))
                    summary_writer.add_summary(summary_str, i)

                del batch_data
                del batch_labels
                del batch_labels_weights

            test_data = extract_data(TEST_SOURCE_DATA, list_images=list_test_data, verbose=0)
            test_labels, test_labels_weights = extract_label(TEST_LABELS_DATA, segmentation_image_size, list_test_labels, verbose=0)
            test_labels = numpy.reshape(test_labels, [test_labels.shape[0] * test_labels.shape[1] * test_labels.shape[2], NUM_LABELS])
            test_data_node = tf.constant(test_data)
            test_prediction = tf.nn.softmax(unet.model(test_data_node))
            # Finally print the result!
            test_error = error_rate(test_prediction.eval(), test_labels)
            sct.printv('Test error: ' + str(test_error))
            timer_training.printTotalTime()

            #savePredictions(result_test_prediction, output_path, list_test_data, segmentation_image_size)

            save_path = saver.save(s, output_path + 'model.ckpt')
            sct.printv('Model saved in file: ' + save_path)


def applySegmentationML(fname_model):
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, fname_model)
        sct.printv('Model restored.')
        # Do some work with the model


if __name__ == '__main__':
    tf.app.run()
