from __future__ import absolute_import

import os
import sys

import pytest

import nibabel as nib

import numpy as np

import keras.backend as K

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))

from spinalcordtoolbox.deepseg_gm import model as gm_model
from spinalcordtoolbox.deepseg_gm import deepseg_gm as gm_core



class TestModel(object):
    """This class will test the model module from deepseg_gm."""

    def test_dice_coef_max(self):
        """Test the upper-bound of the dice coefficient."""
        y_true = np.ones((2, 1, 200, 200))
        y_pred = np.ones((2, 1, 200, 200))
        var_y_true = K.variable(y_true)
        var_y_pred = K.variable(y_pred)
        out_loss = gm_model.dice_coef(var_y_true, var_y_pred)
        res = K.eval(out_loss)
        assert res == 1.0

    def test_dice_coef_min(self):
        """Test the lower-bound of the dice coefficient."""
        y_true = np.ones((2, 1, 200, 200))
        y_pred = np.zeros((2, 1, 200, 200))
        var_y_true = K.variable(y_true)
        var_y_pred = K.variable(y_pred)
        out_loss = gm_model.dice_coef(var_y_true, var_y_pred)
        res = K.eval(out_loss)
        # Smoothing term makes it never reach zero
        assert res == pytest.approx(0.0, abs=0.001)

    def test_dice_loss(self):
        """Test the loss itself, should be negative of upper-bound."""
        y_true = np.ones((2, 1, 200, 200))
        y_pred = np.ones((2, 1, 200, 200))
        var_y_true = K.variable(y_true)
        var_y_pred = K.variable(y_pred)
        out_loss = gm_model.dice_coef_loss(var_y_true, var_y_pred)
        res = K.eval(out_loss)
        assert res == -1.0

    def test_create_model(self):
        """Test the model creation with 32 and 64 filter p/ layer."""
        model = gm_model.create_model(32)
        assert model.count_params() == 127585
        model = gm_model.create_model(64)
        assert model.count_params() == 478017

        diff_size_model = gm_model.create_model(32, (103, 102))
        axial_slices_mock = np.random.randn(1, 103, 102, 1)
        preds = diff_size_model.predict(axial_slices_mock, batch_size=8)
        assert preds.shape == axial_slices_mock.shape


class TestCore(object):
    def test_data_resource(self):
        """Test the DataResource manager, and check if files exists."""
        resource_models = gm_core.DataResource('deepseg_gm_models')
        for model_name in gm_model.MODELS.keys():
            model_path, metadata_path = gm_model.MODELS[model_name]
            metadata_abs_path = resource_models.get_file_path(metadata_path)
            assert os.path.isfile(metadata_abs_path)

    def test_crop_center(self):
        """Test the cropping method, with even and odd sizes."""
        dummy_img = np.ones((203, 202))
        cropped_img, cropped_region = gm_core.crop_center(dummy_img, 200, 200)
        assert cropped_img.shape == (200, 200)
        pad_image = cropped_region.pad(cropped_img)
        assert pad_image.shape == (203, 202)

    def test_thresholding(self):
        """Test thresholding with above and below use cases."""
        dummy_preds = np.full((200, 200), 0.9)
        thr_ret = gm_core.threshold_predictions(dummy_preds, 0.5)
        assert np.count_nonzero(thr_ret) == 200 * 200

        dummy_preds = np.full((200, 200), 0.4)
        thr_ret = gm_core.threshold_predictions(dummy_preds, 0.5)
        assert np.count_nonzero(thr_ret) == 0

        dummy_preds = np.full((200, 200), 0.4)
        thr_ret = gm_core.threshold_predictions(dummy_preds, None)
        assert np.allclose(dummy_preds, thr_ret)

    def test_segment_volume(self):
        """Call the segmentation routine itself with a dummy input."""
        np_data = np.ones((200, 200, 2), dtype=np.float32)
        img = nib.Nifti1Image(np_data, np.eye(4))
        ret = gm_core.segment_volume(img, 'challenge')
        assert ret.shape == (200, 200, 2)

    def test_standardization_transform(self):
        """Test the standardization transform with specified parameters."""
        np_data = np.ones((200, 200, 2), dtype=np.float32)
        np_data[..., 0] = 5.0
        transform = gm_core.StandardizationTransform(3.0, 2.0)
        assert transform.mean == 3.0
        assert transform.std == 2.0

        np_transformed_data = transform(np_data)
        assert np_transformed_data.mean() == 0.0
        assert np_transformed_data.std() == 1.0

    def test_volume_standardization_transform(self):
        """Test the standardization transform with estimated parameters."""
        np_data = np.ones((200, 200, 2), dtype=np.float32)
        np_data[..., 0] = 5.0
        transform = gm_core.VolumeStandardizationTransform()
        np_transformed_data = transform(np_data)
        assert np_transformed_data.mean() == 0.0
        assert np_transformed_data.std() == 1.0
