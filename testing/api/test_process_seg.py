# pytest unit tests for spinalcordtoolbox.process_seg

# TODO: add test with known angle (i.e. not found with fitting)
# TODO: test empty slices and slices with two objects

import pytest
import math
import copy
from datetime import datetime
from random import uniform
import numpy as np
import nibabel as nib
from skimage.transform import rotate

from spinalcordtoolbox import process_seg
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.resampling import resample_nib

# Define global variables
VERBOSE = 0  # set to 2 to save files
DEBUG = False  # Set to True to save images


dict_test_orientation = [
    {'input': 0.0, 'expected': 0.0},
    {'input': math.pi, 'expected': 0.0},
    {'input': -math.pi, 'expected': 0.0},
    {'input': math.pi / 2, 'expected': 90.0},
    {'input': -math.pi / 2, 'expected': 90.0},
    {'input': 2 * math.pi, 'expected': 0.0},
    {'input': math.pi / 4, 'expected': 45.0},
    {'input': -math.pi / 4, 'expected': 45.0},
    {'input': 3 * math.pi / 4, 'expected': 45.0},
    {'input': -3 * math.pi / 4, 'expected': 45.0},
    {'input': math.pi / 8, 'expected': 22.5},
    {'input': -math.pi / 8, 'expected': 22.5},
    {'input': 3 * math.pi / 8, 'expected': 67.5},
    {'input': -3 * math.pi / 8, 'expected': 67.5},
    ]


def dummy_segmentation(size_arr=(256, 256, 256), pixdim=(1, 1, 1), dtype=np.float64, orientation='LPI',
                       shape='rectangle', angle_RL=0, angle_AP=0, angle_IS=0, radius_RL=5.0, radius_AP=3.0,
                       degree=2, interleaved=False, zeroslice=[], debug=False):
    """Create a dummy Image with a ellipse or ones running from top to bottom in the 3rd dimension, and rotate the image
    to make sure that compute_csa and compute_shape properly estimate the centerline angle.
    :param size_arr: tuple: (nx, ny, nz)
    :param pixdim: tuple: (px, py, pz)
    :param dtype: Numpy dtype.
    :param orientation: Orientation of the image. Default: LPI
    :param shape: {'rectangle', 'ellipse'}
    :param angle_RL: int: angle around RL axis (in deg)
    :param angle_AP: int: angle around AP axis (in deg)
    :param angle_IS: int: angle around IS axis (in deg)
    :param radius_RL: float: 1st radius. With a, b = 50.0, 30.0 (in mm), theoretical CSA of ellipse is 4712.4
    :param radius_AP: float: 2nd radius
    :param degree: int: degree of polynomial fit
    :param interleaved: bool: create a dummy segmentation simulating interleaved acquisition
    :param zeroslice: list int: zero all slices listed in this param
    :param debug: Write temp files for debug
    :return: img: Image object
    """
    # Initialization
    padding = 15  # Padding size (isotropic) to avoid edge effect during rotation
    # Create a 3d array, with dimensions corresponding to x: RL, y: AP, z: IS
    nx, ny, nz = [int(size_arr[i] * pixdim[i]) for i in range(3)]
    data = np.zeros((nx, ny, nz))
    xx, yy = np.mgrid[:nx, :ny]

    # Create a dummy segmentation using polynomial function
    # create regularized curve, within Y-Z plane (A-P), located at x=nx/2:
    x = [round(nx / 2.)] * len(range(nz))
    # and passing through the following points:
    # y = np.array([round(ny / 4.), round(ny / 2.), round(3 * ny / 4.)])  # oblique curve (changing AP points across SI)
    y = [round(ny / 2.), round(ny / 2.), round(ny / 2.)]               # straight curve (same location of AP across SI)
    z = np.array([0, round(nz / 2.), nz - 1])
    # we use poly (instead of bspline) in order to allow change of scalar for each term of polynomial function
    p = np.polynomial.Polynomial.fit(z, y, deg=degree)

    # create two polynomial fits, by choosing random scalar for each term of both polynomial functions and then
    # interleave these two fits (one for odd slices, second one for even slices)
    if interleaved:
        p_even = copy.copy(p)
        p_odd = copy.copy(p)
        # choose random scalar for each term of polynomial function
        # even slices
        p_even.coef = [element * uniform(0.5, 1) for element in p_even.coef]
        # odd slices
        p_odd.coef = [element * uniform(0.5, 1) for element in p_odd.coef]
        # performs two polynomial fits - one will serve for even slices, second one for odd slices
        yfit_even = np.round(p_even(range(nz)))
        yfit_odd = np.round(p_odd(range(nz)))

        # combine even and odd polynomial fits
        yfit = np.zeros(nz)
        yfit[0:nz:2] = yfit_even[0:nz:2]
        yfit[1:nz:2] = yfit_odd[1:nz:2]
    # IF INTERLEAVED=FALSE, perform only one polynomial fit without modification of term's scalars
    else:
        yfit = np.round(p(range(nz)))   # has to be rounded for correct float -> int conversion in next step

    yfit = yfit.astype(int)
    # loop across slices and add object
    for iz in range(nz):
        if shape == 'rectangle':  # theoretical CSA: (a*2+1)(b*2+1)
            data[:, :, iz] = ((abs(xx - x[iz]) <= radius_RL) & (abs(yy - yfit[iz]) <= radius_AP)) * 1
        if shape == 'ellipse':
            data[:, :, iz] = (((xx - x[iz]) / radius_RL) ** 2 + ((yy - yfit[iz]) / radius_AP) ** 2 <= 1) * 1

    # Pad to avoid edge effect during rotation
    data = np.pad(data, padding, 'reflect')

    # ROTATION ABOUT IS AXIS
    # rotate (in deg), and re-grid using linear interpolation
    data_rotIS = rotate(data, angle_IS, resize=False, center=None, order=1, mode='constant', cval=0, clip=False,
                        preserve_range=False)

    # ROTATION ABOUT RL AXIS
    # Swap x-z axes (to make a rotation within y-z plane, because rotate will apply rotation on the first 2 dims)
    data_rotIS_swap = data_rotIS.swapaxes(0, 2)
    # rotate (in deg), and re-grid using linear interpolation
    data_rotIS_swap_rotRL = rotate(data_rotIS_swap, angle_RL, resize=False, center=None, order=1, mode='constant',
                                   cval=0, clip=False, preserve_range=False)
    # swap back
    data_rotIS_rotRL = data_rotIS_swap_rotRL.swapaxes(0, 2)

    # ROTATION ABOUT AP AXIS
    # Swap y-z axes (to make a rotation within x-z plane)
    data_rotIS_rotRL_swap = data_rotIS_rotRL.swapaxes(1, 2)
    # rotate (in deg), and re-grid using linear interpolation
    data_rotIS_rotRL_swap_rotAP = rotate(data_rotIS_rotRL_swap, angle_AP, resize=False, center=None, order=1,
                                         mode='constant', cval=0, clip=False, preserve_range=False)
    # swap back
    data_rot = data_rotIS_rotRL_swap_rotAP.swapaxes(1, 2)

    # Crop image (to remove padding)
    data_rot_crop = data_rot[padding:nx+padding, padding:ny+padding, padding:nz+padding]

    # Zero specified slices
    if zeroslice is not []:
        data_rot_crop[:, :, zeroslice] = 0

    # Create nibabel object
    xform = np.eye(4)
    for i in range(3):
        xform[i][i] = 1  # in [mm]
    nii = nib.nifti1.Nifti1Image(data_rot_crop.astype('float32'), xform)
    # resample to desired resolution
    nii_r = resample_nib(nii, new_size=pixdim, new_size_type='mm', interpolation='linear')
    # Create Image object. Default orientation is LPI.
    # For debugging add .save() at the end of the command below
    img = Image(np.asanyarray(nii_r.dataobj), hdr=nii_r.header, dim=nii_r.header.get_data_shape())
    # Update orientation
    img.change_orientation(orientation)
    if debug:
        img.save('tmp_dummy_seg_'+datetime.now().strftime("%Y%m%d%H%M%S%f")+'.nii.gz')
    return img


@pytest.mark.parametrize('test_orient', dict_test_orientation)
def test_fix_orientation(test_orient):
    assert process_seg.fix_orientation(test_orient['input']) == pytest.approx(test_orient['expected'], rel=0.0001)


# Generate a list of fake segmentation for testing: (dummy_segmentation(params), dict of expected results)
im_segs = [
    # test area
    (dummy_segmentation(size_arr=(32, 32, 5), debug=DEBUG),
     {'area': 77, 'angle_RL': 0.0, 'angle_AP': 0.0, 'length': 5.0},
     {'angle_corr': False}),
    # test anisotropic pixel dim
    (dummy_segmentation(size_arr=(64, 32, 5), pixdim=(0.5, 1, 5), debug=DEBUG),
     {'area': 77, 'angle_RL': 0.0, 'angle_AP': 0.0},
     {'angle_corr': False}),
    # test with angle IS
    (dummy_segmentation(size_arr=(32, 32, 5), pixdim=(1, 1, 5), angle_IS=15, debug=DEBUG),
     {'area': 77, 'angle_RL': 0.0, 'angle_AP': 0.0},
     {'angle_corr': False}),
    # test with ellipse shape
    (dummy_segmentation(size_arr=(64, 64, 5), shape='ellipse', radius_RL=13.0, radius_AP=5.0, angle_RL=0.0,
                        debug=DEBUG),
     {'area': 197.0, 'diameter_AP': 10.0, 'diameter_RL': 26.0, 'angle_RL': 0.0, 'angle_AP': 0.0},
     {'angle_corr': False}),
    # test with int16. Different bit ordering, which can cause issue when applying transform.warp()
    (dummy_segmentation(size_arr=(64, 320, 5), pixdim=(1, 1, 1), dtype=np.int16, orientation='RPI',
                        shape='rectangle', radius_RL=13.0, radius_AP=5.0, angle_RL=0.0, debug=DEBUG),
     {'area': 297.0, 'angle_RL': 0.0, 'angle_AP': 0.0},
     {'angle_corr': False}),
    # test with angled spinal cord (neg angle)
    (dummy_segmentation(size_arr=(64, 64, 20), shape='ellipse', radius_RL=13.0, radius_AP=5.0, angle_RL=-30.0,
                        debug=DEBUG),
     {'area': 197.0, 'diameter_AP': 10.0, 'diameter_RL': 26.0, 'angle_RL': -30.0, 'angle_AP': 0.0, 'length': 23.15},
     {'angle_corr': True}),
    # test with AP angled spinal cord
    (dummy_segmentation(size_arr=(64, 64, 20), shape='ellipse', radius_RL=13.0, radius_AP=5.0, angle_AP=20.0,
                        debug=DEBUG),
     {'area': 197.0, 'diameter_AP': 9.41, 'diameter_RL': 26.0, 'angle_RL': 0.0, 'angle_AP': 20.0, 'length': 21.02},
     {'angle_corr': True}),
    # test with RL and AP angled spinal cord
    (dummy_segmentation(size_arr=(64, 64, 50), shape='ellipse', radius_RL=13.0, radius_AP=5.0,
                        angle_RL=-10.0, angle_AP=15.0, debug=DEBUG),
     {'area': 196.0, 'diameter_AP':  9.465, 'diameter_RL': 26.0, 'angle_RL': -10.0, 'angle_AP': 15.0},
     {'angle_corr': True}),
    # Reproduce issue: "LinAlgError: SVD did not converge". Note: due to the cropping, the estimated angle_RL is wrong,
    # so it had to be made wrong in the expected values
    (dummy_segmentation(size_arr=(64, 64, 50), shape='ellipse', radius_RL=13.0, radius_AP=5.0,
                        angle_RL=-10.0, angle_AP=30.0, debug=DEBUG),
     {'area': 197.0, 'diameter_AP': 9.418, 'diameter_RL': 26.0, 'angle_RL': -11.5, 'angle_AP': 30.0},
     {'angle_corr': True}),
    # test uint8 input
    (dummy_segmentation(size_arr=(32, 32, 50), dtype=np.uint8, angle_RL=15, debug=DEBUG),
     {'area': 77, 'angle_RL': 13.269097801736525, 'angle_AP': 0.0},
     {'angle_corr': True}),
    # test all output params
    (dummy_segmentation(size_arr=(128, 128, 5), pixdim=(1, 1, 1), shape='ellipse', radius_RL=50.0, radius_AP=30.0,
                        debug=DEBUG),
     {'area': 4701, 'angle_AP': 0.0, 'angle_RL': 0.0, 'diameter_AP': 60.0, 'diameter_RL': 100.0, 'eccentricity': 0.8,
      'orientation': 0.0},
     {'angle_corr': False}),
    # test with one empty slice
    (dummy_segmentation(size_arr=(32, 32, 5), zeroslice=[2], debug=DEBUG),
     {'area': np.nan},
     {'angle_corr': False, 'slice': 2})
    ]


@pytest.mark.parametrize('im_seg,expected,params', im_segs)
def test_compute_shape(im_seg, expected, params):
    metrics, fit_results = process_seg.compute_shape(im_seg,
                                                     angle_correction=params['angle_corr'],
                                                     param_centerline=ParamCenterline(),
                                                     verbose=VERBOSE)
    for key in expected.keys():
        # fetch obtained_value
        if 'slice' in params:
            obtained_value = float(metrics['area'].data[params['slice']])
        else:
            if key == 'length':
                # when computing length, sums values across slices
                obtained_value = metrics[key].data.sum()
            else:
                # otherwise, average across slices
                obtained_value = metrics[key].data.mean()
        # fetch expected_value
        if expected[key] is np.nan:
            assert math.isnan(obtained_value)
            break
        else:
            expected_value = pytest.approx(expected[key], rel=0.05)
        assert obtained_value == expected_value
