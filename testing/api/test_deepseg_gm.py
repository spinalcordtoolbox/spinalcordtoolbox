import os

import nibabel as nib
import numpy as np

from spinalcordtoolbox.deepseg_gm import deepseg_gm as gm_core


class TestCore(object):
    def test_data_resource(self):
        """Test the DataResource manager, and check if files exists."""
        resource_models = gm_core.DataResource('deepseg_gm_models')
        for model_name in gm_core.MODELS.keys():
            model_path, metadata_path = gm_core.MODELS[model_name]
            metadata_abs_path = resource_models.get_file_path(metadata_path)
            assert os.path.isfile(metadata_abs_path)

    def test_shape_transformation(self):
        """Test the cropping/padding method with even and odd sizes."""
        dim_new = (200, 200)
        for dim_old in ((200, 200), (203, 202), (198, 197)):
            dummy_img = np.ones(dim_old)
            shape_transform = gm_core.ShapeTransform(dummy_img.shape, dim_new)
            transformed_image = shape_transform.apply(dummy_img)
            assert transformed_image.shape == dim_new
            inverse_image = shape_transform.undo(transformed_image)
            assert inverse_image.shape == dim_old

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
        for dim in ((200, 200, 3), (203, 202, 2), (198, 197, 1)):
            np_data = np.ones(dim, dtype=np.float32)
            img = nib.Nifti1Image(np_data, np.eye(4))
            ret = gm_core.segment_volume(img, 'challenge')
            assert ret.shape == dim

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
