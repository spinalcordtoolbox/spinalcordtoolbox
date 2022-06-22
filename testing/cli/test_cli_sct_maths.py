import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import sct_test_path
from spinalcordtoolbox.scripts import sct_maths

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_maths_percent_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_maths.main(argv=['-i', 'mt/mtr.nii.gz', '-percent', '95', '-o', 'test.nii.gz'])


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_maths_add_integer_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_maths.main(argv=['-i', 'mt/mtr.nii.gz', '-add', '1', '-o', 'test.nii.gz'])


@pytest.mark.parametrize('dim', ['0', '1', '2'])
def test_sct_maths_symmetrize(dim, tmp_path):
    """Run the CLI script, then verify that symmetrize properly flips and
    averages the image data."""
    path_in = sct_test_path('t2', 't2.nii.gz')
    path_out = str(tmp_path/f't2_sym_{dim}.nii.gz')
    sct_maths.main(argv=['-i', path_in, '-symmetrize', str(dim),
                         '-o', path_out])
    im_in = Image(path_out)
    im_out = Image(path_out)
    assert np.array_equal(im_out.data,
                          (im_in.data + np.flip(im_in.data, axis=int(dim))) / 2.0)


def run_arithmetic_operation(tmp_path, dims, op):
    """Generate some dummy data, then run -add/-sub/-mul/-div on the data."""
    args = []
    # Add input images to argument list
    for i, dim in enumerate(dims):
        path_im = str(tmp_path / f"im_{i}.nii.gz")
        Image(np.ones(dim)).save(path_im)
        if i == 0:
            args += ["-i", path_im]
            args += [op]
        else:
            args += [path_im]
    # Add output image to argument list
    path_out = str(tmp_path / f"im_out{op}.nii.gz")
    args += ["-o", path_out]
    # Call sct_maths and return output data
    sct_maths.main(args)
    return Image(path_out).data


@pytest.mark.parametrize('ndims', [(3, 3), (4, 4), (3, 3, 3), (4, 4, 4)])
@pytest.mark.parametrize('op', ['-add', '-mul'])
def test_add_mul_output_dimensions(tmp_path, ndims, op):
    """Test that '-add' and '-mul' return the correct dimensions across various combinations
       of 3D and 4D images."""
    possible_dims = {3: [10, 15, 20], 4: [10, 15, 20, 5]}
    dims = [possible_dims[n] for n in ndims]
    data_out = run_arithmetic_operation(tmp_path, dims, op)
    # For `-add` and `-mul`, output dimensions should match the minimum ndim
    # - 3D {+,*} 3D = 3D
    # - 4D {+,*} 4D = 4D
    # - 3D {+,*} 4D = 3D  (i.e. we add/mul all the 3D volumes within the 4D image)
    # - 4D {+,*} 3D = 3D  (i.e. we add/mul all the 3D volumes within the 4D image)
    dim_expected = possible_dims[min(ndims)]
    dim_out = list(data_out.shape)
    assert dim_out == dim_expected, f"Calling {op} on ndims {ndims} resulted in mismatch."


# NB: -sub/-div do not currently support mis-matched 3D/4D ndims:
#      ('ndims', [(3, 4), (4, 3), (3, 4, 4), (3, 4, 3), (4, 3, 3), (4, 4, 3)])
@pytest.mark.parametrize('ndims', [(3, 3), (4, 4), (3, 3, 3), (4, 4, 4)])
@pytest.mark.parametrize('op', ['-sub', '-div'])
def test_sub_div_output_dimensions(op, ndims, tmp_path):
    """Test that '-sub' and '-div' return the correct dimensions across various combinations
       of 3D and 4D images."""
    possible_dims = {3: [10, 15, 20], 4: [10, 15, 20, 5]}
    dims = [possible_dims[n] for n in ndims]
    data_out = run_arithmetic_operation(tmp_path, dims, op)
    # For `-sub` and `-div`, output dimensions should match the dimensions of the input image
    # - 3D {-,/} 3D = 3D
    # - 4D {-,/} 4D = 4D
    dim_expected = dims[0]
    dim_out = list(data_out.shape)
    assert dim_out == dim_expected, f"Calling {op} on ndims {ndims} resulted in mismatch."


@pytest.mark.parametrize('op', ['-add', '-mul'])
def test_add_mul_4d_image_with_no_argument(op, tmp_path):
    """Test that passing a 4D image with bare '-add' or '-mul' operates on the 3D volumes."""
    # Generate input image
    base_dim = [20, 20, 20]
    n_vol = 5
    dim = base_dim + [n_vol]
    path_im = str(tmp_path / "im.nii.gz")
    val = 2
    Image(np.ones(dim) * val).save(path_im)
    # Generate output image
    path_out = str(tmp_path / "im_out.nii.gz")
    sct_maths.main(["-i", path_im, op, "-o", path_out])
    # Validate output data
    data_out = Image(path_out).data
    dim_out = list(data_out.shape)
    assert dim_out == base_dim
    if op == '-add':
        assert np.all(data_out == val * n_vol)
    elif op == '-mul':
        assert np.all(data_out == val ** n_vol)
