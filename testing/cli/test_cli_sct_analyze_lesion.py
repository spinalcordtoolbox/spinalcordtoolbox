# pytest unit tests for sct_analyze_lesion
import os.path
import shutil

import pytest
import logging

import math
import pickle
import numpy as np

from scipy.ndimage import center_of_mass

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.fs import extract_fname
from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.scripts import sct_analyze_lesion, sct_label_utils, sct_register_to_template, sct_warp_template

logger = logging.getLogger(__name__)


@pytest.fixture()
def dummy_lesion(request, tmp_path, tmp_path_qc):
    """Define fake voxel lesions using the specified dimensions."""
    lesion_params = request.param
    if isinstance(lesion_params, list) and all(isinstance(t, tuple) for t in lesion_params):
        lesion_params = [lesion_params]  # If we only have a single lesion, encapsulate in a list to iterate

    # Create the list of coordinates spanning the lesions
    coordinates = []
    for (starting_coord, dim) in lesion_params:
        for x in range(starting_coord[0], starting_coord[0] + dim[0]):
            for y in range(starting_coord[1], starting_coord[1] + dim[1]):
                for z in range(starting_coord[2], starting_coord[2] + dim[2]):
                    coord = [str(x), str(y), str(z), "1"]
                    coordinates.append(",".join(coord))
    # Format the coordinates into a str argument that `-create` can accept
    create_arg = ":".join(coordinates)

    # Create the lesion mask file and output to a temporary directory
    path_ref = sct_test_path("t2", "t2.nii.gz")
    path_out = str(tmp_path/"lesion.nii.gz")
    sct_label_utils.main(argv=['-i', path_ref, '-o', path_out,
                               '-create', create_arg,
                               '-qc', tmp_path_qc])

    return path_out, lesion_params


def compute_expected_measurements(lesion_params, path_seg=None):
    """
    Compute the expected measurements from the provided lesion dimensions.
    :param lesion_params: list of tuples, each containing the starting coordinates (x, y, z) and dimensions (width,
    height, depth) of a dummy lesion
    :param path_seg: str, path to the spinal cord segmentation
    :return: dict, expected measurements for each lesion
    """
    if path_seg:
        # Get center of mass of the largest lesion in S-I axis; AIL --> [1]
        # Note: as the spinal cord curvature is not too big, we can simply use the lesion coordinates; i.e., no need
        # to use center-of-mass of the lesion
        largest_lesion = lesion_params[0]  # largest lesion is always the first one in the list
        # Manually remove (crop) the lesion slice (z slice 19) outside the spinal cord segmentation to pass the test.
        # This also means that we need to adjust the depth from 5 to 4.
        # In the actual `sct_analyze_lesion` script, this is done by array multiplication with the spinal cord, but here
        # we use lesion's starting coordinates and dimensions
        if largest_lesion[0] == (29, 40, 19):
            largest_lesion = [(29, 40, 20), (3, 3, 4)]
        z_center = int(round(np.mean(list(range(largest_lesion[0][1], largest_lesion[0][1] + largest_lesion[1][1])))))
        z_range = np.arange(z_center - 2, z_center + 3)  # two slices above and below the lesion center of mass
        # For each of these slices, compute the spinal cord center of mass in the L-R direction
        sc_com = [center_of_mass(Image(path_seg).data[:, z, :])[1] for z in
                  z_range]  # 2D slice from AIL: [1] --> L-R
        mid_sagittal_slice = np.mean(sc_com)  # target slice in right-left axis (x direction) for the interpolation

    # Loop over lesions
    expected_measurements_dict = dict()
    for idx, (starting_coord, dim) in enumerate(lesion_params):
        # Manually remove (crop) the lesion slice (z slice 19) outside the spinal cord segmentation to pass the test.
        # This also means that we need to adjust the depth from 5 to 4.
        # In the actual `sct_analyze_lesion` script, this is done by array multiplication with the spinal cord, but here
        # we use lesion's starting coordinates and dimensions
        if starting_coord == (29, 40, 19):
            starting_coord, dim = [(29, 40, 20), (3, 3, 4)]
        if path_seg:
            # Find the minimum SC area surrounding the lesion
            data_seg = Image(path_seg).data
            min_area = min(np.sum(data_seg[:, n_slice, :])
                           for n_slice in range(starting_coord[1], starting_coord[1] + dim[1]))

            # Find the minimum mid-sagittal tissue bridge width for each LR slice in the lesion
            x = starting_coord[0] + (dim[0] // 2)  # Compute midpoint of lesion (to split into dorsal/ventral regions)
            tissue_bridges = {}
            for z in range(starting_coord[2], starting_coord[2] + dim[2]):

                # for each SI slice in the lesion, compute the bridge widths
                dorsal_bridge_widths, ventral_bridge_widths = [], []
                for y in range(starting_coord[1], starting_coord[1] + dim[1]):
                    # compute ventral widths
                    ventral_sc_width = np.sum(data_seg[:x, y, z])
                    ventral_lesion_width = (x - starting_coord[0])
                    ventral_bridge_widths.append(max(0.0, ventral_sc_width - ventral_lesion_width))
                    # compute dorsal widths
                    dorsal_sc_width = np.sum(data_seg[x:, y, z])
                    dorsal_lesion_width = (starting_coord[0] + dim[0] - x)
                    dorsal_bridge_widths.append(max(0.0, dorsal_sc_width - dorsal_lesion_width))

                # find minimum widths
                tissue_bridges[f"slice_{z}_dorsal_bridge_width [mm]"] = min(dorsal_bridge_widths)
                tissue_bridges[f"slice_{z}_ventral_bridge_width [mm]"] = min(ventral_bridge_widths)
                tissue_bridges[f"slice_{z}_total_bridge_width [mm]"] = (min(dorsal_bridge_widths) +
                                                                        min(ventral_bridge_widths))
        else:
            mid_sagittal_slice = None
            min_area = 0
            tissue_bridges = {}

        # Compute the expected (voxel) measurements from the provided dimensions
        # NB: Actual measurements will differ slightly due to spine curvature
        measurements = {
            # NB: 'sct_analyze_lesion' treats lesions as cylinders. So:
            #   - Vertical axis: Length of the cylinder
            'length [mm]': dim[1],
            'width [mm]': dim[0],
            'interpolated_midsagittal_slice': mid_sagittal_slice,
            # NB: we can compute length_midsagittal_slice and width_midsagittal_slice here from dim for the purposes of
            #  testing, but in the actual script, we need the spinal cord segmentation to compute these values based on
            #  the midsagittal slice
            'length_interpolated_midsagittal_slice [mm]': dim[1],
            'width_interpolated_midsagittal_slice [mm]': dim[0],
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
            'max_axial_damage_ratio []': dim[0] * dim[2] / min_area if min_area != 0 else None,
            **tissue_bridges
        }

    expected_measurements_dict[idx] = measurements

    return expected_measurements_dict


@pytest.mark.sct_testing
# Each tuple represents the starting coordinates in the AIL orientation (x: AP, y: IS, z: LR)
# and dimensions (width, height, depth) of a dummy lesion
@pytest.mark.parametrize("dummy_lesion, rtol", [
    # Straight region of `t2.nii.gz` -> little curvature -> smaller tolerance
    ([(29, 45, 25), (3, 10, 2)], 0.001),
    ([(29, 27, 25), (1, 4, 1)], 0.001),  # NB: Example from #3633
    # Curved region of `t2.nii.gz` -> lots of curvature -> larger tolerance
    ([(31, 0, 25), (4, 15, 3)], 0.1),
    # Multiple lesions
    ([[(31, 0, 25), (4, 15, 3)],
      [(29, 45, 25), (3, 10, 2)]], 0.1),
    # Lesion partly outside the spinal cord segmentation (z (LR) slice 19 is outside the SC seg)
    ([(29, 40, 19), (3, 3, 5)], 0.001)
], indirect=["dummy_lesion"])
def test_sct_analyze_lesion_matches_expected_dummy_lesion_measurements(dummy_lesion, rtol, tmp_path, tmp_path_qc):
    """Run the CLI script and verify that the lesion measurements match
    expected values."""
    # Run the analysis on the dummy lesion file
    path_lesion, lesion_params = dummy_lesion
    path_seg = sct_test_path("t2", "t2_seg-manual.nii.gz")
    sct_analyze_lesion.main(argv=['-m', path_lesion,
                                  '-s', path_seg,
                                  '-ofolder', str(tmp_path),
                                  '-qc', tmp_path_qc])

    # Test presence of output files
    _, fname, _ = extract_fname(path_lesion)
    for suffix in ['_analysis.pkl', '_analysis.xlsx', '_label.nii.gz']:
        assert os.path.isfile(tmp_path / f"{fname}{suffix}")

    # Load analysis results from pickled pandas.Dataframe
    with open(tmp_path/f"{fname}_analysis.pkl", 'rb') as f:
        measurements = pickle.load(f)['measures']

    # generate 1 measurement dict per lesion
    expected_measurements_dict = compute_expected_measurements(lesion_params, path_seg)

    for idx, expected_measurements in expected_measurements_dict.items():
        # Validate analysis results
        for key, expected_value in expected_measurements.items():
            # For interpolated measures, because the midsagittal slice is computed using the **spinal cord** center of
            # mass, there is no guarantee that this interpolated slice will fully contain the lesion. Meaning, the
            # averaged value may be computed using empty slices, and thus it should be either close to or less than the
            # expected value.
            if key in ['length_interpolated_midsagittal_slice [mm]', 'width_interpolated_midsagittal_slice [mm]']:
                try:
                    np.testing.assert_allclose(measurements.at[idx, key], expected_value, rtol=rtol)
                except AssertionError as e:
                    if not (measurements.at[idx, key] <= expected_value):
                        raise e  # Only raise exception if the value is greater than expected
            # These measures are the same regardless of angle adjustment/spine curvature
            elif key in ['volume [mm3]', 'max_axial_damage_ratio []']:
                np.testing.assert_equal(measurements.at[idx, key], expected_value)
            else:
                # However, these measures won't match exactly due to angle adjustment
                # from spinal cord centerline curvature
                np.testing.assert_allclose(measurements.at[idx, key],
                                           expected_value, rtol=rtol)
                # The values will be adjusted according to the cos of the angle
                # between the spinal cord centerline and the S-I axis, as per:
                # https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3681#discussion_r804822552
                if key == 'max_equivalent_diameter [mm]' or 'bridge' in key:
                    assert measurements.at[idx, key] <= expected_value
                elif key == 'length [mm]':
                    assert measurements.at[idx, key] >= expected_value


@pytest.mark.sct_testing
# Each tuple represents the starting coordinates (x, y, z) and dimensions (width, height, depth) of a dummy lesion
@pytest.mark.parametrize("dummy_lesion, rtol", [
    # Straight region of `t2.nii.gz` -> little curvature -> smaller tolerance
    ([(29, 45, 25), (3, 10, 2)], 0.001),
    ([(29, 27, 25), (1, 4, 1)], 0.001),  # NB: Example from #3633
    # Curved region of `t2.nii.gz` -> lots of curvature -> larger tolerance
    ([(31, 0, 25), (4, 15, 3)], 0.01),
    # Multiple lesions
    ([[(31, 0, 25), (4, 15, 3)],
      [(29, 45, 25), (3, 10, 2)]], 0.01)
], indirect=["dummy_lesion"])
def test_sct_analyze_lesion_matches_expected_dummy_lesion_measurements_without_segmentation(dummy_lesion, rtol,
                                                                                            tmp_path, tmp_path_qc):
    """Run the CLI script without providing SC segmentation -- only volume is computed. Max_equivalent_diameter and
    length are nan."""
    # Run the analysis on the dummy lesion file
    path_lesion, lesion_params = dummy_lesion
    sct_analyze_lesion.main(argv=['-m', path_lesion,
                                  '-ofolder', str(tmp_path),
                                  '-qc', tmp_path_qc])  # A warning will be printed because no SC seg

    # Test presence of output files
    _, fname, _ = extract_fname(path_lesion)
    for suffix in ['_analysis.pkl', '_analysis.xlsx', '_label.nii.gz']:
        assert os.path.isfile(tmp_path / f"{fname}{suffix}")

    # Load analysis results from pickled pandas.Dataframe
    with open(tmp_path/f"{fname}_analysis.pkl", 'rb') as f:
        measurements = pickle.load(f)['measures']

    # generate 1 measurement dict per lesion
    expected_measurements_dict = compute_expected_measurements(lesion_params)

    # Compute expected measurements from the lesion dimensions
    for idx, expected_measurements in expected_measurements_dict.items():
        # Validate analysis results
        for key, expected_value in expected_measurements.items():
            if key == 'volume [mm3]':
                np.testing.assert_equal(measurements.at[idx, key], expected_value)
            # The max_equivalent_diameter, length, and damage ratio are nan because no segmentation is provided
            elif key in ['max_equivalent_diameter [mm]', 'length [mm]', 'max_axial_damage_ratio []']:
                assert math.isnan(measurements.at[idx, key])


@pytest.mark.parametrize("dummy_lesion", [
    ([(29, 0, 25), (4, 15, 3)])
], indirect=["dummy_lesion"])
def test_sct_analyze_lesion_with_template(dummy_lesion, tmp_path, tmp_path_qc):
    # prep the template for use with `-f` argument of sct_analyze_lesion
    sct_register_to_template.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                                        '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                        '-l', sct_test_path('t2', 'labels.nii.gz'),
                                        '-t', sct_test_path('template'),
                                        '-ofolder', str(tmp_path),
                                        '-qc', tmp_path_qc])
    sct_warp_template.main(argv=['-d', sct_test_path('t2', 't2.nii.gz'),
                                 '-w', str(tmp_path/'warp_template2anat.nii.gz'),
                                 '-a', '0',  # -a is '1' by default, but small template doesn't have atlas
                                 '-t', sct_test_path('template'),
                                 '-ofolder', str(tmp_path),
                                 '-qc', tmp_path_qc])
    template_path = tmp_path / 'template'
    shutil.copy(template_path / "PAM50_small_levels.nii.gz",
                template_path / "PAM50_levels.nii.gz")  # Rename to comply with sct_analyze_lesion expectations
    (tmp_path / 'atlas').mkdir()  # make a dummy atlas folder to avoid errors due to expected folder

    # Run the analysis on the dummy lesion file
    path_lesion, _ = dummy_lesion
    sct_analyze_lesion.main(argv=['-m', path_lesion,
                                  '-f', str(tmp_path),
                                  '-ofolder', str(tmp_path),
                                  '-qc', tmp_path_qc])
    _, fname, _ = extract_fname(path_lesion)
    for suffix in ['_analysis.pkl', '_analysis.xlsx', '_label.nii.gz']:
        assert os.path.isfile(tmp_path / f"{fname}{suffix}")
