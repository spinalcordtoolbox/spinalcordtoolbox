import os
import logging

import pytest
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import extract_fname
from spinalcordtoolbox.scripts import sct_image

logger = logging.getLogger(__name__)


@pytest.fixture
def dmri_in():
    """Filepath for input dmri image."""
    return 'dmri/dmri.nii.gz'


@pytest.fixture
def dmri_t_slices(tmp_path, dmri_in):
    """Filepaths for dmri image split across t axis."""
    fname_out = str(tmp_path / 'dmri.nii.gz')
    sct_image.main(argv=['-i', dmri_in, '-split', 't', '-o', fname_out])

    parent, filename, ext = extract_fname(fname_out)
    fname_slices = [os.path.join(parent, filename + '_T' + str(i).zfill(4) + ext) for i in range(7)]
    return fname_slices


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_image_pad():
    """Run the CLI script and ."""
    pad = 2
    path_in = 'mt/mtr.nii.gz'
    path_out = 'sct_image_out.nii.gz'
    sct_image.main(argv=['-i', path_in, '-o', 'sct_image_out.nii.gz', '-pad', f'0,0,{pad}'])

    nx, ny, nz, nt, px, py, pz, pt = Image(path_in).dim
    nx2, ny2, nz2, nt2, px2, py2, pz2, pt2 = Image(path_out).dim
    assert nz2 == nz + 2 * pad


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize("path_in", ['t2/t2.nii.gz', 'dmri/dmri.nii.gz'])
def test_sct_image_getorient(path_in):
    """Run the CLI script and ."""
    sct_image.main(argv=['-i', path_in, '-getorient'])


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_image_split_dmri(dmri_t_slices):
    """Verify the output of '-split' matches reference image. Note: CLI script is run by the 'dmri_t_slices' fixture."""
    _, filename, ext = extract_fname(dmri_t_slices[0])
    ref = Image(f'dmri / {filename}{ext}')  # Reference image should exist inside working directory (sct_testing_data)
    new = Image(dmri_t_slices[0])         # New image should be generated inside tmp directory
    assert np.sum(ref.data - new.data) <= 1e-3


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_image_concat(tmp_path, dmri_t_slices, dmri_in):
    """Run the CLI script and verify concatenated imaeg matches reference image."""
    path_out = str(tmp_path / 'dmri_concat.nii.gz')
    sct_image.main(argv=['-i'] + dmri_t_slices + ['-concat', 't', '-o', path_out])
    ref = Image(dmri_in)
    new = Image(path_out)
    assert np.sum(ref.data - new.data) <= 1e-3
