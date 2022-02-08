import pytest
import logging

import pickle
import numpy as np

from spinalcordtoolbox.utils import sct_test_path, extract_fname
from spinalcordtoolbox.scripts import sct_analyze_lesion, sct_label_utils

logger = logging.getLogger(__name__)


@pytest.fixture()
def dummy_lesion(tmp_path):
    """Define a fake voxel lesion using the specified dimensions."""
    # TODO: Parametrize this fixture to test other lesion dimensions
    starting_coord = [29, 27, 25]
    dim = [1, 4, 1]

    # Format the coordinates into a str argument that `-create` can accept
    coordinates = []
    for x in range(starting_coord[0], starting_coord[0] + dim[0]):
        for y in range(starting_coord[1], starting_coord[1] + dim[1]):
            for z in range(starting_coord[2], starting_coord[2] + dim[2]):
                coord = [str(x), str(y), str(z), "1"]
                coordinates.append(",".join(coord))
    create_arg = ":".join(coordinates)

    # Create the lesion mask file and output to a temporary directory
    path_ref = sct_test_path("t2", "t2.nii.gz")
    path_out = str(tmp_path/"lesion.nii.gz")
    sct_label_utils.main(argv=['-i', path_ref, '-o', path_out,
                               '-create', create_arg])

    # Compute the expected measurements from the provided dimensions
    measurements = {
        # NB: 'sct_analyze_lesion' treats lesions as cylinders. So:
        #   - Y axis: Length of the cylinder
        'length [mm]': dim[1],
        #   - X/Z plane: Cross-sectional slices of the cylinder.
        #                Specifically, 'max_equivalent_diameter' takes the X/Z
        #                cross-sectional area of the lesion (which is computed
        #                using square voxels), then finds the diameter of an
        #                equivalent *circle* with that same area:
        #                   a = pi*r^2
        #                -> a = pi*(d/2)^2
        #                -> d = 2*sqrt(a/pi)
        'max_equivalent_diameter [mm]': 2 * np.sqrt(dim[0] * dim[2] / np.pi),
        'volume [mm3]': dim[0] * dim[1] * dim[2],
    }

    return path_out, measurements


@pytest.mark.sct_testing
def test_sct_analyze_lesion_output_file_exists(dummy_lesion, tmp_path):
    """Run the CLI script and verify output file exists."""
    # Run the analysis on the dummy lesion file
    path_lesion, expected_measurements = dummy_lesion
    sct_analyze_lesion.main(argv=['-m', path_lesion,
                                  '-s', sct_test_path("t2", "t2_seg-manual.nii.gz"),
                                  '-ofolder', str(tmp_path)])

    # Load analysis results from pickled pandas.Dataframe
    _, fname, _ = extract_fname(path_lesion)
    with open(tmp_path/f"{fname}_analyzis.pkl", 'rb') as f:
        measurements = pickle.load(f)['measures']

    # Validate analysis results
    for key, expected_value in expected_measurements.items():
        if key == 'volume [mm3]':
            np.testing.assert_equal(measurements.at[0, key], expected_value)
        else:
            # The length/diameter won't match exactly due to spine curvature
            np.testing.assert_allclose(measurements.at[0, key],
                                       expected_value, rtol=0.01)
