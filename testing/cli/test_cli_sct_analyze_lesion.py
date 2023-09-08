# pytest unit tests for sct_analyze_lesion

import pytest
import logging

import math
import pickle
import numpy as np

from spinalcordtoolbox.image import Image
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

    return path_out, starting_coord, dim


def compute_expected_measurements(dim, starting_coord=None, path_seg=None):
    # Find the minimum SC area surrounding the lesion
    if path_seg:
        data_seg = Image(path_seg).data
        min_area = min(np.sum(data_seg[:, n_slice, :])
                       for n_slice in range(starting_coord[1], starting_coord[1] + dim[1]))

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
        # Take the dummy lesion CSA and divide it by the minimum surrounding SC seg area
        # NB: We should account for voxel resolution, but in this case it's just 1.0mm/voxel
        'max_axial_damage_ratio []': dim[0] * dim[2] / min_area if path_seg else None
    }

    return measurements


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
    path_lesion, starting_coord, dim = dummy_lesion
    path_seg = sct_test_path("t2", "t2_seg-manual.nii.gz")
    sct_analyze_lesion.main(argv=['-m', path_lesion,
                                  '-s', path_seg,
                                  '-ofolder', str(tmp_path)])

    # Load analysis results from pickled pandas.Dataframe
    _, fname, _ = extract_fname(path_lesion)
    with open(tmp_path/f"{fname}_analysis.pkl", 'rb') as f:
        measurements = pickle.load(f)['measures']

    # Compute expected measurements from the lesion dimensions
    expected_measurements = compute_expected_measurements(dim, starting_coord, path_seg)

    # Validate analysis results
    for key, expected_value in expected_measurements.items():
        # These measures are the same regardless of angle adjustment/spine curvature
        if key in ['volume [mm3]', 'max_axial_damage_ratio []']:
            np.testing.assert_equal(measurements.at[0, key], expected_value)
        else:
            # However, these measures won't match exactly due to angle adjustment
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


@pytest.mark.sct_testing
@pytest.mark.parametrize("dummy_lesion, rtol", [
    # Straight region of `t2.nii.gz` -> little curvature -> smaller tolerance
    ([(29, 45, 25), (3, 10, 2)], 0.001),
    ([(29, 27, 25), (1, 4, 1)], 0.001),  # NB: Example from #3633
    # Curved region of `t2.nii.gz` -> lots of curvature -> larger tolerance
    ([(29, 0, 25), (4, 15, 3)], 0.01)
], indirect=["dummy_lesion"])
def test_sct_analyze_lesion_matches_expected_dummy_lesion_measurements_without_segmentation(dummy_lesion, rtol,
                                                                                            tmp_path):
    """Run the CLI script without providing SC segmentation -- only volume is computed. Max_equivalent_diameter and
    length are nan."""
    # Run the analysis on the dummy lesion file
    path_lesion, _, dim = dummy_lesion
    sct_analyze_lesion.main(argv=['-m', path_lesion,
                                  '-ofolder', str(tmp_path)])

    # Load analysis results from pickled pandas.Dataframe
    _, fname, _ = extract_fname(path_lesion)
    with open(tmp_path/f"{fname}_analysis.pkl", 'rb') as f:
        measurements = pickle.load(f)['measures']

    # Compute expected measurements from the lesion dimensions
    expected_measurements = compute_expected_measurements(dim)

    # Validate analysis results
    for key, expected_value in expected_measurements.items():
        if key == 'volume [mm3]':
            np.testing.assert_equal(measurements.at[0, key], expected_value)
        # The max_equivalent_diameter, length, and damage ratio are nan because no segmentation is provided
        elif key in ['max_equivalent_diameter [mm]', 'length [mm]', 'max_axial_damage_ratio []']:
            assert math.isnan(measurements.at[0, key])
