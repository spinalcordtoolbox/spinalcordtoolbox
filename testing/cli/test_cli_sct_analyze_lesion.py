import pytest
import logging

import pickle
import numpy as np

from spinalcordtoolbox.utils import sct_test_path, extract_fname
from spinalcordtoolbox.scripts import sct_analyze_lesion, sct_label_utils

logger = logging.getLogger(__name__)


@pytest.fixture()
def dummy_lesion(request, tmp_path):
    """Define a fake voxel lesion using the specified dimensions."""
    starting_coord, dim = request.param

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

    # Compute the expected (voxel) measurements from the provided dimensions
    # NB: Actual measurements will differ slightly due to spine curvature
    measurements = {
        # NB: 'sct_analyze_lesion' treats lesions as cylinders. So:
        #   - Vertical axis: Length of the cylinder
        'length [mm]': dim[1],
        #   - Horizontal plane: Cross-sectional slices of the cylinder.
        #        Specifically, 'max_equivalent_diameter' takes the
        #        cross-sectional area of the lesion (which is computed
        #        using square voxels), then finds the diameter of an
        #        equivalent *circle* with that same area:
        #           a = pi*r^2
        #        -> a = pi*(d/2)^2
        #        -> d = 2*sqrt(a/pi)
        'max_equivalent_diameter [mm]': 2 * np.sqrt(dim[0] * dim[2] / np.pi),
        'volume [mm3]': dim[0] * dim[1] * dim[2],
    }

    return path_out, measurements


@pytest.mark.sct_testing
@pytest.mark.parametrize("dummy_lesion, rtol", [
    # Straight region of `t2.nii.gz` -> little curvature -> smaller tolerance
    ([(29, 45, 25), (3, 10, 2)], 0.001),
    ([(29, 27, 25), (1, 4, 1)], 0.001),  # NB: Example from #3633
    # Curved region of `t2.nii.gz` -> lots of curvature -> larger tolerance
    ([(29, 0, 25), (4, 15, 3)], 0.01)
], indirect=["dummy_lesion"])
def test_sct_analyze_lesion_matches_expected_dummy_lesion_measurements(dummy_lesion, rtol, tmp_path):
    """Run the CLI script and verify that the lesion measurements match
    expected values."""
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
            # The length/diameter won't match exactly due to angle adjustment
            # from spinal cord centerline curvature
            np.testing.assert_allclose(measurements.at[0, key],
                                       expected_value, rtol=rtol)
            # The values will be adjusted according to the cos of the angle
            # between the spinal cord centerline and the S-I axis, as per:
            # https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3681#discussion_r804822552
            if key == 'max_equivalent_diameter [mm]':
                assert measurements.at[0, key] < expected_value
            elif key == 'length [mm]':
                assert measurements.at[0, key] > expected_value
