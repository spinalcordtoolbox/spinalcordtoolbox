import os

import pytest
import logging
import numpy as np  # noqa: F401

from spinalcordtoolbox.image import Image  # noqa: F401
from spinalcordtoolbox.utils import sct_test_path, sct_dir_local_path
from spinalcordtoolbox.scripts import sct_register_multimodal, sct_create_mask

logger = logging.getLogger(__name__)


def test_sct_register_multimodal_mask_files_exist(tmp_path):
    """
    Run the script without validating results.

    - TODO: Write a check that verifies the registration results.
    - TODO: Parametrize this test to add '-initwarpinv warp_anat2template.nii.gz',
            after the file is added to sct_testing_data:
            https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3407#discussion_r646895013
    """
    fname_mask = str(tmp_path/'mask_mt1.nii.gz')
    sct_create_mask.main(['-i', sct_test_path('mt', 'mt1.nii.gz'),
                          '-p', f"centerline,{sct_test_path('mt', 'mt1_seg.nii.gz')}",
                          '-size', '35mm', '-f', 'cylinder', '-o', fname_mask])
    sct_register_multimodal.main([
        '-i', sct_dir_local_path('data/PAM50/template/', 'PAM50_t2.nii.gz'),
        '-iseg', sct_dir_local_path('data/PAM50/template/', 'PAM50_cord.nii.gz'),
        '-d', sct_test_path('mt', 'mt1.nii.gz'),
        '-dseg', sct_test_path('mt', 'mt1_seg.nii.gz'),
        '-param', 'step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3',
        '-m', fname_mask,
        '-initwarp', sct_test_path('t2', 'warp_template2anat.nii.gz'),
        '-ofolder', str(tmp_path)
    ])

    for path in ["PAM50_t2_reg.nii.gz", "warp_PAM50_t22mt1.nii.gz"]:
        assert os.path.exists(tmp_path/path)

    # Because `-initwarp` was specified (but `-initwarpinv` wasn't) the dest->seg files should NOT exist
    for path in ["mt1_reg.nii.gz", "warp_mt12PAM50_t2.nii.gz"]:
        assert not os.path.exists(tmp_path/path)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize("use_seg,param,fname_gt", [
    (False, 'step=1,algo=syn,type=im,iter=1,smooth=1,shrink=2,metric=MI', 'mt/mt0_reg_syn_goldstandard.nii.gz'),
    (False, 'step=1,algo=slicereg,type=im,iter=5,smooth=0,metric=MeanSquares', 'mt/mt0_reg_slicereg_goldstandard.nii.gz'),
    (True, 'step=1,algo=centermassrot,type=seg,rot_method=pca', None),
    (True, 'step=1,algo=centermassrot,type=imseg,rot_method=hog', None),
    (True, 'step=1,algo=centermassrot,type=imseg,rot_method=pcahog', None),
    (True, 'step=1,algo=columnwise,type=seg,smooth=1', None),
])
def test_sct_register_multimodal_mt0_image_data_within_threshold(use_seg, param, fname_gt):
    """Run the CLI script and verify that the output image data is close to a reference image (within threshold)."""
    fname_out = 'mt0_reg.nii.gz'

    argv = ['-i', 'mt/mt0.nii.gz', '-d', 'mt/mt1.nii.gz', '-o', fname_out, '-x', 'linear', '-r', '0', '-param', param]
    seg_argv = ['-iseg', 'mt/mt0_seg.nii.gz', '-dseg', 'mt/mt1_seg.nii.gz']
    sct_register_multimodal.main(argv=(argv + seg_argv) if use_seg else argv)

    # This check is skipped because of https://github.com/neuropoly/spinalcordtoolbox/issues/3372
    #############################################################################################
    # if fname_gt is not None:
    #     im_gt = Image(fname_gt)
    #     im_result = Image(fname_out)
    #     # get dimensions
    #     nx, ny, nz, nt, px, py, pz, pt = im_gt.dim
    #     # set the difference threshold to 1e-3 pe voxel
    #     threshold = 1e-3 * nx * ny * nz * nt
    #     # check if non-zero elements are present when computing the difference of the two images
    #     diff = im_gt.data - im_result.data
    #     # compare images
    #     assert abs(np.sum(diff)) < threshold  # FIXME: Use np.linalg.norm when this test is fixed
