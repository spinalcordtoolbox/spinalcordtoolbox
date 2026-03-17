# pytest unit tests for sct_dmri_moco

import os

import numpy
import numpy as np
import pytest
import logging

from numpy import allclose, genfromtxt

from spinalcordtoolbox.scripts import sct_dmri_moco, sct_image, sct_crop_image, sct_create_mask, sct_deepseg, sct_apply_transfo
from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def dmri_mean_seg(tmp_path_factory):
    """Mean segmented image for QC report generation."""
    tmp_path = tmp_path_factory.mktemp('dmri_mean_seg')
    path_out = str(tmp_path / 'dmri_mean_seg.nii.gz')
    sct_deepseg.main(argv=['spinalcord', '-i', sct_test_path('dmri', 'dwi_mean.nii.gz'),
                           '-o', path_out, '-qc', str(tmp_path)])
    return path_out


@pytest.mark.sct_testing
def test_sct_dmri_moco_check_params(tmp_path, tmp_path_qc, dmri_mean_seg):
    """Run the CLI script and validate output moco params."""
    sct_dmri_moco.main(argv=['-i', sct_test_path('dmri', 'dmri.nii.gz'),
                             '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                             '-g', '3', '-x', 'nn', '-r', '0',
                             '-ofolder', str(tmp_path),
                             '-qc', tmp_path_qc, '-qc-seg', dmri_mean_seg])

    lresults = genfromtxt(tmp_path / "moco_params.tsv", skip_header=1, delimiter='\t')
    lgroundtruth = [
        0.001201150922494186,
        3.276041445156287e-05,
        3.276041445156287e-05,
        3.276041445156287e-05,
        0.2046662087725081,
        0.2046662087725081,
        0.2046662087725081,
    ]
    assert allclose(lresults, lgroundtruth)


@pytest.fixture
def dmri_mask(tmp_path):
    """Mask image for testing."""
    path_out = str(tmp_path / 'mask.nii')
    sct_create_mask.main(argv=['-i', sct_test_path('dmri', 'dmri_T0000.nii.gz'),
                               '-p', 'coord,21x17', '-size', '15mm', '-f', 'gaussian',
                               '-o', path_out])

    return path_out


@pytest.mark.sct_testing
def test_sct_dmri_moco_with_mask_check_params(tmp_path, dmri_mask, tmp_path_qc, dmri_mean_seg):
    """Run the CLI script with '-m' option and validate output moco params."""
    sct_dmri_moco.main(argv=['-i', sct_test_path('dmri', 'dmri.nii.gz'),
                             '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                             '-g', '3', '-r', '0',
                             '-m', dmri_mask, '-ofolder', str(tmp_path),
                             '-qc', tmp_path_qc, '-qc-seg', dmri_mean_seg])

    lresults = genfromtxt(tmp_path / "moco_params.tsv", skip_header=1, delimiter='\t')
    lgroundtruth = [
        0.01207862,
        0.01163640,
        0.01163640,
        0.01163640,
        0.16997247,
        0.16997247,
        0.16997247,
    ]
    assert allclose(lresults, lgroundtruth)


def build_disp_from_params(fname_tx, fname_ty, fname_ref, fname_warp, orientation):
    """Re-build displacement field from translation parameters; Tx and Ty"""
    tx = Image(fname_tx).data[0, 0, :, :]  # (Z,T)
    ty = Image(fname_ty).data[0, 0, :, :]  # (Z,T)
    ref = Image(fname_ref)
    disp = np.zeros(list(ref.data.shape) + [3], dtype=np.float32)
    nt = disp.shape[3]
    if orientation == "RPI":
        nz = disp.shape[2]
        for t in range(nt):
            for z in range(nz):
                disp[:, :, z, t, 0] = tx[z, t]
                disp[:, :, z, t, 1] = ty[z, t]
                disp[:, :, z, t, 2] = 0.0
    elif orientation == "SAL":
        nz = disp.shape[0]
        for t in range(nt):
            for z in range(nz):
                # FIXME: There is a problem with the moco params output for non-RPI images. moco_x/moco_y can't be used
                #        to reconstruct the warping field in the same way as for RPI images. See PR.
                disp[z, :, :, t, 0] = tx[z, t] * -1
                disp[z, :, :, t, 1] = ty[z, t] * -1
                disp[z, :, :, t, 2] = 0.0
    else:
        # This is a quick hack for speed of prototyping, and probably shouldn't be merged as-is
        raise NotImplementedError("Test is not yet generalized to orientations outside RPI/SAL")
    im_disp = Image(disp, hdr=ref.hdr)
    im_disp.affine = ref.affine
    im_disp.hdr.set_data_shape(disp.shape)
    im_disp.hdr.set_intent('vector', (), '')
    im_disp.save(fname_warp)


@pytest.mark.sct_testing
@pytest.mark.parametrize("orientation", ["SAL"])
def test_sct_dmri_moco_dl(tmp_path, dmri_mask, tmp_path_qc, dmri_mean_seg, orientation):
    """Run the CLI script with '-m' and '-ref' option and using '-dl' algorithm."""
    # reorient input images according to orientation parameter
    inputs = {
        '-i': sct_test_path('dmri', 'dmri.nii.gz'),
        '-ref': sct_test_path('dmri', 'dwi_mean.nii.gz'),
        '-m': dmri_mask,
        '-qc-seg': dmri_mean_seg
    }
    for arg, im_in in inputs.items():
        im_out = str(tmp_path / add_suffix(os.path.basename(im_in), f"_{orientation}"))
        im = Image(im_in)
        im.change_orientation(orientation)
        im.save(im_out)
        inputs[arg] = im_out

    # 1. apply moco_dl directly (outputs motion-corrected image + Tx/Ty moco params)
    sct_dmri_moco.main(argv=['-i', inputs['-i'], '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                             '-ref', inputs['-ref'], '-m', inputs['-m'],
                             '-ofolder', str(tmp_path), '-dl',
                             '-qc', tmp_path_qc, '-qc-seg', inputs['-qc-seg']])
    fname_moco = add_suffix(inputs['-i'], '_mocoDL')

    # 2. build warping field from Tx/Ty and apply it to the original image using sct_apply_transfo
    fname_warp_4d = str(tmp_path / "warp_dmri_moco.nii.gz")
    build_disp_from_params(fname_tx=str(tmp_path / "moco_params_x.nii.gz"),
                           fname_ty=str(tmp_path / "moco_params_y.nii.gz"),
                           fname_ref=inputs['-i'], fname_warp=fname_warp_4d,
                           orientation=orientation)
    # split both the warping field and original input image into 3D volumes
    sct_image.main(argv=['-i', inputs['-i'], '-split', 't', '-o', str(tmp_path / "dmri.nii.gz")])
    sct_image.main(argv=['-i', fname_warp_4d, '-split', 't'])
    # warp a single volume using the corresponding 3D warping field
    t = 0
    fname_warp_3d = str(tmp_path / f"warp_dmri_moco_T{t:04d}.nii.gz")
    fname_vol_in = str(tmp_path / f"dmri_T{t:04d}.nii.gz")
    fname_vol_out = str(tmp_path / f"dmri_warped_T{t:04d}.nii.gz")
    sct_apply_transfo.main(argv=['-i', fname_vol_in, '-d', sct_test_path('dmri', 'dwi_mean.nii.gz'),
                                 '-w', fname_warp_3d, '-o', fname_vol_out, '-x', 'linear'])

    # 3. compare a single slice from the two t=0 volumes from both methods using phase cross-correlation
    from skimage.registration import phase_cross_correlation

    warped_vol0 = Image(fname_vol_out).change_orientation('RPI').data.astype(np.float32)
    mocodl_vol0 = Image(fname_moco).change_orientation('RPI').data.astype(np.float32)[..., 0]
    z = warped_vol0.shape[2] // 2  # pick a representative slice: middle slice
    shift, _, _ = phase_cross_correlation(warped_vol0[:, :, z],
                                          mocodl_vol0[:, :, z],
                                          normalization=None, upsample_factor=10)
    assert np.max(np.abs(shift)) < 1.0


@pytest.fixture
def dmri_ail_cropped(tmp_path):
    """Reorient image to sagittal for testing another orientation (and crop to save time)."""
    path_out_orient = str(tmp_path / 'dmri_AIL.nii')
    path_out = str(tmp_path / 'dmri_AIL_crop.nii')
    sct_image.main(argv=['-i', sct_test_path('dmri', 'dmri.nii.gz'),
                         '-setorient', 'AIL', '-o', path_out_orient])
    sct_crop_image.main(argv=['-i', path_out_orient, '-zmin', '19', '-zmax', '21', '-o', path_out])

    return path_out


@pytest.mark.sct_testing
def test_sct_dmri_moco_sagittal_no_checks(tmp_path, tmp_path_qc, dmri_mean_seg, dmri_ail_cropped):
    """Run the CLI script, but don't check anything."""
    sct_dmri_moco.main(argv=['-i', dmri_ail_cropped, '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                             '-x', 'nn', '-r', '0',
                             '-ofolder', str(tmp_path), '-qc', tmp_path_qc, '-qc-seg', dmri_mean_seg])
    # NB: We skip checking params because there are no output moco params for sagittal images (*_AIL)


@pytest.mark.parametrize("group_size", [-1, 0, 1.5, 'NaN'])
def test_sct_dmri_moco_invalid_group_values(tmp_path, tmp_path_qc, dmri_mean_seg, group_size):
    """Ensure that invalid group sizes return a parsing error."""
    with pytest.raises(SystemExit) as e:
        sct_dmri_moco.main(argv=['-i', sct_test_path('dmri', 'dmri.nii.gz'),
                                 '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                                 '-g', str(group_size), '-x', 'nn', '-r', '0',
                                 '-ofolder', str(tmp_path),
                                 '-qc', tmp_path_qc, '-qc-seg', dmri_mean_seg])
    assert e.value.code == 2
