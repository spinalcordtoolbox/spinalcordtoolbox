import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_register_multimodal

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize("use_seg,param", [
    (False, 'step=1,algo=syn,type=im,iter=1,smooth=1,shrink=2,metric=MI'),
    (False, 'step=1,algo=slicereg,type=im,iter=5,smooth=0,metric=MeanSquares'),
    (True, 'step=1,algo=centermassrot,type=seg,rot_method=pca'),
    (True, 'step=1,algo=centermassrot,type=imseg,rot_method=hog'),
    (True, 'step=1,algo=centermassrot,type=imseg,rot_method=pcahog'),
    (True, 'step=1,algo=columnwise,type=seg,smooth=1'),
])
def test_sct_register_multimodal_mt0_image_data_within_threshold(use_seg, param):
    """Run the CLI script and verify that the output image data is close to a reference image (within threshold)."""
    fname_out = 'mt0_reg.nii.gz'
    fname_gt = None

    argv = ['-i', 'mt/mt0.nii.gz', '-d', 'mt/mt1.nii.gz', '-o', fname_out, '-x', 'linear', '-r', '0', '-param', param]
    seg_argv = ['-iseg', 'mt/mt0_seg.nii.gz', '-dseg', 'mt/mt1_seg.nii.gz']
    sct_register_multimodal.main(argv=(argv + seg_argv) if use_seg else argv)

    # FIXME: The 'sct_testing' version of this test never did anything ("--> N/A"), because 'fname_gt' was never defined
    #  Need to figure out what fname_gt should be so we can properly make this comparison.
    if fname_gt is not None:
        im_gt = Image(fname_gt)
        im_result = Image(fname_out)
        # get dimensions
        nx, ny, nz, nt, px, py, pz, pt = im_gt.dim
        # set the difference threshold to 1e-3 pe voxel
        threshold = 1e-3 * nx * ny * nz * nt
        # check if non-zero elements are present when computing the difference of the two images
        diff = im_gt.data - im_result.data
        # compare images
        assert abs(np.sum(diff)) < threshold
