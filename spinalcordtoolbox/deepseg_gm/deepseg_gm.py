# coding: utf-8
# This is the interface API for the deepseg_gm model
# that implements the model for the Spinal Cord Gray Matter Segmentation.
#
# Reference paper:
#     Perone, C. S., Calabrese, E., & Cohen-Adad, J. (2017).
#     Spinal cord gray matter segmentation using deep dilated convolutions.
#     URL: https://arxiv.org/abs/1710.01269

from __future__ import absolute_import, print_function

import warnings
import json
import os
import sys
import io

import nibabel as nib
import numpy as np

# Avoid Keras logging
original_stderr = sys.stderr
if sys.hexversion < 0x03000000:
    sys.stderr = io.BytesIO()
else:
    sys.stderr = io.TextIOWrapper(io.BytesIO(), sys.stderr.encoding)
try:
    from keras import backend as K
except Exception as e:
    sys.stderr = original_stderr
    raise
else:
    sys.stderr = original_stderr

from spinalcordtoolbox import resampling
from . import model
from ..utils import __data_dir__


# Suppress warnings and TensorFlow logging
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SMALL_INPUT_SIZE = 200
BATCH_SIZE = 4


def check_backend():
    """This function will check for the current backend and
    then it will warn the user if the backend is theano."""
    if K.backend() != 'tensorflow':
        print("\nWARNING: you're using a Keras backend different than\n"
              "Tensorflow, which is not recommended. Please verify\n"
              "your configuration file according to: https://keras.io/backend/\n"
              "to make sure you're using Tensorflow Keras backend.\n")
    return K.backend()


class DataResource(object):
    """This class is responsible for resource file
    management (such as loding models)."""

    def __init__(self, dirname):
        """Initialize the resource with the directory
        name context.

        :param dirname: the root directory name.
        """
        self.data_root = os.path.join(__data_dir__, dirname)

    def get_file_path(self, filename):
        """Get the absolute file path based on the
        data root directory.

        :param filename: the filename.
        """
        return os.path.join(self.data_root, filename)


class CroppedRegion(object):
    """This class holds cropping information about the volume
    center crop.
    """

    def __init__(self, original_shape, starts, crops):
        """Constructor for the CroppedRegion.

        :param original_shape: the original volume shape.
        :param starts: crop beginning (x, y).
        :param crops: the crops (x, y).
        """
        self.originalx = original_shape[0]
        self.originaly = original_shape[1]
        self.startx = starts[0]
        self.starty = starts[1]
        self.cropx = crops[0]
        self.cropy = crops[1]

    def pad(self, image):
        """This method will pad an image using the saved
        cropped region.

        :param image: the image to pad.
        :return: padded image.
        """
        bef_x = self.startx
        aft_x = self.originalx - (self.startx + self.cropx)

        bef_y = self.starty
        aft_y = self.originaly - (self.starty + self.cropy)

        padded = np.pad(image,
                        ((bef_y, aft_y),
                         (bef_x, aft_x)),
                        mode="constant")
        return padded


class StandardizationTransform(object):
    """This transformation will standardize the volume
    according to the specified mean/std.dev.
    """
    def __init__(self, mean, std):
        """Constructor for the normalization transformation.

        :param mean: the mean parameter
        :param std: the standar deviation parameter
        """
        self.mean = mean
        self.std = std

    def __call__(self, volume):
        """This method will enable the function call for the
        class object.

        :param volume: the volume to be normalized.
        """
        volume -= self.mean
        volume /= self.std
        return volume


class VolumeStandardizationTransform(object):
    """This transformation will standardize the volume with
    the parameters estimated from the volume itself.
    """

    def __call__(self, volume):
        """This method will enable the function call for the
        class object.

        :param volume: the volume to be normalized.
        """
        volume_mean = volume.mean()
        volume_std = volume.std()

        volume -= volume_mean
        volume /= volume_std

        return volume


def crop_center(img, cropx, cropy):
    """This function will crop the center of the volume image.

    :param img: image to be cropped.
    :param cropx: x-coord of the crop.
    :param cropy: y-coord of the crop.
    :return: (cropped image, cropped region)
    """
    y, x = img.shape

    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)

    if startx < 0 or starty < 0:
        raise RuntimeError("Negative crop.")

    cropped_region = CroppedRegion((x, y), (startx, starty),
                                   (cropx, cropy))

    return img[starty:starty + cropy,
               startx:startx + cropx], cropped_region


def threshold_predictions(predictions, thr=0.999):
    """This method will threshold predictions.

    :param thr: the threshold (if None, no threshold will
                be applied).
    :return: thresholded predictions
    """
    if thr is None:
        return predictions[:]

    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds <= thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds > thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


def segment_volume(ninput_volume, model_name,
                   threshold=0.999, use_tta=False):
    """Segment a nifti volume.

    :param ninput_volume: the input volume.
    :param model_name: the name of the model to use.
    :param threshold: threshold to be applied in predictions.
    :param use_tta: whether TTA (test-time augmentation)
                    should be used or not.
    :return: segmented slices.
    """
    gmseg_model_challenge = DataResource('deepseg_gm_models')
    model_path, metadata_path = model.MODELS[model_name]

    metadata_abs_path = gmseg_model_challenge.get_file_path(metadata_path)
    with open(metadata_abs_path) as fp:
        metadata = json.load(fp)

    volume_size = np.array(ninput_volume.shape[0:2])
    small_input = (volume_size <= SMALL_INPUT_SIZE).any()

    if small_input:
        # Smaller than the trained net, don't crop
        net_input_size = volume_size
    else:
        # larger sizer, crop at 200x200
        net_input_size = (SMALL_INPUT_SIZE, SMALL_INPUT_SIZE)

    deepgmseg_model = model.create_model(metadata['filters'],
                                         net_input_size)

    model_abs_path = gmseg_model_challenge.get_file_path(model_path)
    deepgmseg_model.load_weights(model_abs_path)

    volume_data = ninput_volume.get_data()
    axial_slices = []
    crops = []

    for slice_num in range(volume_data.shape[2]):
        data = volume_data[..., slice_num]

        if not small_input:
            data, cropreg = crop_center(data, SMALL_INPUT_SIZE,
                                        SMALL_INPUT_SIZE)
            crops.append(cropreg)

        axial_slices.append(data)

    axial_slices = np.asarray(axial_slices, dtype=np.float32)
    axial_slices = np.expand_dims(axial_slices, axis=3)

    normalization = VolumeStandardizationTransform()
    axial_slices = normalization(axial_slices)

    if use_tta:
        pred_sampled = []
        for i in range(8):
            sampled_value = np.random.uniform(high=2.0)
            sampled_axial_slices = axial_slices + sampled_value
            preds = deepgmseg_model.predict(sampled_axial_slices,
                                            batch_size=BATCH_SIZE,
                                            verbose=True)
            pred_sampled.append(preds)

        preds = deepgmseg_model.predict(axial_slices, batch_size=BATCH_SIZE,
                                        verbose=True)
        pred_sampled.append(preds)
        pred_sampled = np.asarray(pred_sampled)
        pred_sampled = np.mean(pred_sampled, axis=0)
        preds = threshold_predictions(pred_sampled, threshold)
    else:
        preds = deepgmseg_model.predict(axial_slices, batch_size=BATCH_SIZE,
                                        verbose=True)
        preds = threshold_predictions(preds, threshold)

    pred_slices = []

    # Un-cropping
    for slice_num in range(preds.shape[0]):
        pred_slice = preds[slice_num][..., 0]
        if not small_input:
            pred_slice = crops[slice_num].pad(pred_slice)
        pred_slices.append(pred_slice)

    pred_slices = np.asarray(pred_slices, dtype=np.uint8)
    pred_slices = np.transpose(pred_slices, (1, 2, 0))

    return pred_slices


def segment_file(input_filename, output_filename,
                 model_name, threshold, verbosity,
                 use_tta):
    """Segment a volume file.

    :param input_filename: the input filename.
    :param output_filename: the output filename.
    :param model_name: the name of model to use.
    :param threshold: threshold to apply in predictions (if None,
                      no threshold will be applied)
    :param verbosity: the verbosity level.
    :param use_tta: whether it should use TTA (test-time augmentation)
                    or not.
    :return: the output filename.
    """
    nii_original = nib.load(input_filename)
    pixdim = nii_original.header["pixdim"][3]
    target_resample = [0.25, 0.25, pixdim]

    nii_resampled = resampling.resample_nib(
        nii_original, new_size=target_resample, new_size_type='mm', interpolation='linear')
    pred_slices = segment_volume(nii_resampled, model_name, threshold,
                                 use_tta)

    original_res = [
        nii_original.header["pixdim"][1],
        nii_original.header["pixdim"][2],
        nii_original.header["pixdim"][3]]

    volume_affine = nii_resampled.affine
    volume_header = nii_resampled.header
    nii_segmentation = nib.Nifti1Image(pred_slices, volume_affine,
                                       volume_header)
    nii_resampled_original = resampling.resample_nib(
        nii_segmentation, new_size=original_res, new_size_type='mm', interpolation='linear')
    res_data = nii_resampled_original.get_data()

    # Threshold after resampling, only if specified
    if threshold is not None:
        res_data = threshold_predictions(res_data, 0.5)

    nib.save(nii_resampled_original, output_filename)
    return output_filename
