import pytest
from dipy.data.fetcher import read_bvals_bvecs

from spinalcordtoolbox.utils import sct_test_path
from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.scripts import (sct_dmri_concat_b0_and_dwi,
                                       sct_dmri_denoise_patch2self,
                                       sct_compute_snr)


@pytest.fixture()
def dmri_single_volumes(tmp_path):
    """
    Create .nii.gz, bvals, and bvecs files for individual dMRI volumes.

    This prep is necessary because `sct_testing_data` lacks individual bvals
    and bvecs files for the `dmri_T000#.nii.gz` files, and
    sct_dmri_separate_b0_and_dwi won't generate new bvals/bvecs file either.
    """
    # Get all bvals/bvecs corresponding to individual dMRI volumes
    bvals, bvecs = read_bvals_bvecs(sct_test_path('dmri', 'bvals.txt'),
                                    sct_test_path('dmri', 'bvecs.txt'))

    fname_dmri_imgs, fname_bvals, fname_bvecs = [], [], []
    for i, (bval, bvec) in enumerate(zip(bvals, bvecs)):
        # 1. Use existing single-volume dMRI files from sct_testing_data
        fname_dmri_imgs.append(sct_test_path('dmri', f'dmri_T000{i}.nii.gz'))

        # 2. Copy bval into single-volume bvals file
        fname_bval = str(tmp_path/f'bvals_T000{i}.txt')
        fname_bvals.append(fname_bval)
        with open(fname_bval, 'w') as f:
            f.write(str(bval))

        # 3. Copy bvec into single-volume bvecs file
        fname_bvec = str(tmp_path/f'bvecs_T000{i}.txt')
        fname_bvecs.append(fname_bvec)
        with open(fname_bvec, 'w') as f:
            f.write(' '.join(map(str, bvec)))

    return fname_dmri_imgs, fname_bvals, fname_bvecs


@pytest.fixture()
def dmri_concat(dmri_single_volumes, tmp_path):
    """
    Concatenate individual dMRI volumes together to create an image with
    >=10 volumes.

    TODO: This is only necessary due to an upstream dipy bug. Once this bug
          is fixed, we can remove this fixture, and simply use `dmri.nii.gz`
          and `bvals.txt` directly.
    """
    # Initialize with b0 image, then add DWI images
    paths_imgs, paths_bvals, paths_bvecs, order = [], [], [], []

    # Duplicating the last volume is the easiest way to get a "realistic"
    # 10+ volume dMRI image with the volumes we have access to.
    for volume_number in ([0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6]):
        order.append('b0' if volume_number == 0 else 'dwi')
        paths_imgs.append(dmri_single_volumes[0][volume_number])
        if volume_number != 0:  # bvals/bvecs only relevant for DWI images
            paths_bvals.append(dmri_single_volumes[1][volume_number])
            paths_bvecs.append(dmri_single_volumes[2][volume_number])

    fname_dmri_concat = str(tmp_path/'dmri_concat.nii.gz')
    fname_bvals_concat = str(tmp_path/'bvals_concat.txt')
    fname_bvecs_concat = str(tmp_path/'bvecs_concat.txt')

    sct_dmri_concat_b0_and_dwi.main(argv=['-i'] + paths_imgs +
                                         ['-bvec'] + paths_bvecs +
                                         ['-bval'] + paths_bvals +
                                         ['-order'] + order +
                                         ['-o', fname_dmri_concat,
                                          '-obval', fname_bvals_concat,
                                          '-obvec', fname_bvecs_concat])

    return fname_dmri_concat, fname_bvals_concat, fname_bvecs_concat


@pytest.mark.parametrize("model", ["ols", "ridge", "lasso"])
@pytest.mark.parametrize("radius", ["0", "1,1,1"])
def test_sct_dmri_denoise_patch2self(dmri_concat, tmp_path, model, radius):
    fname_dmri, fname_bvals, _ = dmri_concat  # dMRI file with >=10 volumes
    fname_out = str(tmp_path/"dmri_denoised.nii.gz")
    sct_dmri_denoise_patch2self.main(argv=['-i', fname_dmri, '-b', fname_bvals,
                                           '-model', model, '-radius', radius,
                                           '-o', fname_out, '-v', '2'])
    snr = []
    for fname in fname_dmri, fname_out:
        sct_compute_snr.main(argv=['-i', fname, '-method', 'mult'])
        img_snr = Image(add_suffix(fname, "_SNR-mult"))
        snr.append(img_snr.data.mean())

    # Mean SNR should increase. Not a thorough test; I just wanted to verify
    # that it even works at all, as the actual underlying method is tested
    # extensively by dipy upstream.
    assert snr[1] > snr[0]
