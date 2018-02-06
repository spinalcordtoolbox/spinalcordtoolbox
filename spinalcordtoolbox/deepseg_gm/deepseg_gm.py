# coding: utf-8
# This is the interface API for the deepseg_gm model
# that implements the model for the Spinal Cord Gray Matter Segmentation.
#
# Reference paper:
#     Perone, C. S., Calabrese, E., & Cohen-Adad, J. (2017).
#     Spinal cord gray matter segmentation using deep dilated convolutions.
#     URL: https://arxiv.org/abs/1710.01269

import warnings
import json
import os
import sys
import io

import nipy
import nibabel as nib
from nipy.io.nifti_ref import nipy2nifti, nifti2nipy
import numpy as np

# Avoid Keras logging
original_stderr = sys.stderr
sys.stderr = io.BytesIO()
from keras import backend as K
sys.stderr = original_stderr

from spinalcordtoolbox.resample import nipy_resample
from . import model


# Suppress warnings and TensorFlow logging
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        path_script = os.path.dirname(__file__)
        directory_level = [".."] * 6 + ["data"]
        data_dir = os.path.join(path_script,
                                *directory_level)

        data_dir = os.path.abspath(data_dir)
        self.data_root = os.path.join(data_dir, dirname)

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

    :param thr: the threshold.
    :return: thresholded predictions.
    """
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds <= thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds > thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


def segment_volume(ninput_volume, model_name):
    """Segment a nifti volume.

    :param ninput_volume: the input volume.
    :param model_name: the name of the model to use.
    :return: segmented slices.
    """
    gmseg_model_challenge = DataResource('deepseg_gm_models')
    model_path, metadata_path = model.MODELS[model_name]

    metadata_abs_path = gmseg_model_challenge.get_file_path(metadata_path)
    with open(metadata_abs_path) as fp:
        metadata = json.load(fp)

    deepgmseg_model = model.create_model(metadata['filters'])

    model_abs_path = gmseg_model_challenge.get_file_path(model_path)
    deepgmseg_model.load_weights(model_abs_path)

    volume_data = ninput_volume.get_data()
    axial_slices = []
    crops = []

    for slice_num in xrange(volume_data.shape[2]):
        data = volume_data[..., slice_num]
        data, cropreg = crop_center(data, model.CROP_HEIGHT,
                                    model.CROP_WIDTH)
        axial_slices.append(data)
        crops.append(cropreg)

    axial_slices = np.asarray(axial_slices, dtype=np.float32)
    axial_slices = np.expand_dims(axial_slices, axis=3)

    axial_slices -= metadata['mean_train']
    axial_slices /= metadata['std_train']

    preds = deepgmseg_model.predict(axial_slices, batch_size=8,
                                    verbose=True)
    preds = threshold_predictions(preds)
    pred_slices = []

    # Un-cropping
    for slice_num in xrange(preds.shape[0]):
        pred_slice = preds[slice_num][..., 0]
        pred_slice = crops[slice_num].pad(pred_slice)
        pred_slices.append(pred_slice)

    pred_slices = np.asarray(pred_slices, dtype=np.uint8)
    pred_slices = np.transpose(pred_slices, (1, 2, 0))

    return pred_slices


def segment_file(input_filename, output_filename,
                 model_name, verbosity):
    """Segment a volume file.

    :param input_filename: the input filename.
    :param output_filename: the output filename.
    :param model_name: the name of model to use.
    :param verbosity: the verbosity level.
    :return: the output filename.
    """
    nii_original = nipy.load_image(input_filename)
    pixdim = nii_original.header["pixdim"][3]
    target_resample = "0.25x0.25x{:.5f}".format(pixdim)

    nii_resampled = nipy_resample.resample_image(nii_original,
                                                 target_resample,
                                                 'mm', 'linear',
                                                 verbosity)

    if (nii_resampled.shape[0] < 200) \
       or (nii_resampled.shape[1] < 200):
        raise RuntimeError("Image too small ({}, {})".format(
                           nii_resampled.shape[0],
                           nii_resampled.shape[1]))

    nii_resampled = nipy2nifti(nii_resampled)
    pred_slices = segment_volume(nii_resampled, model_name)

    original_res = "{:.5f}x{:.5f}x{:.5f}".format(
        nii_original.header["pixdim"][1],
        nii_original.header["pixdim"][2],
        nii_original.header["pixdim"][3])

    volume_affine = nii_resampled.affine
    volume_header = nii_resampled.header
    nii_segmentation = nib.Nifti1Image(pred_slices, volume_affine,
                                       volume_header)
    nii_segmentation = nifti2nipy(nii_segmentation)

    nii_resampled_original = nipy_resample.resample_image(nii_segmentation,
                                                          original_res,
                                                          'mm', 'linear',
                                                          verbosity)

    res_data = nii_resampled_original.get_data()
    res_data = threshold_predictions(res_data, 0.5)

    nipy.save_image(nii_resampled_original, output_filename)

    return output_filename
