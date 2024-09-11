# pytest unit tests for sct_analyze_lesion
import os.path
import shutil

import pytest
import logging

import math
import pickle
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.fs import extract_fname
from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.scripts import sct_analyze_lesion, sct_label_utils, sct_register_to_template, sct_warp_template

logger = logging.getLogger(__name__)


@pytest.fixture()
def dummy_lesion(request, tmp_path):
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
                               '-create', create_arg])

    return path_out, lesion_params


def compute_expected_measurements(dim, starting_coord=None, path_seg=None):
    if path_seg:
        # Find the minimum SC area surrounding the lesion
        data_seg = Image(path_seg).data
        min_area = min(np.sum(data_seg[:, n_slice, :])
                       for n_slice in range(starting_coord[1], starting_coord[1] + dim[1]))

        # Find the midsagittal slice of the spinal cord (assuming AIL input image)
        nonzero_slices = np.unique(np.where(data_seg)[2])  # AIL image: [2] -> LR (sagittal)
        mid_sagittal_slice = int(np.mean([np.min(nonzero_slices), np.max(nonzero_slices)]))

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
        min_area = 0
        tissue_bridges = {}
        mid_sagittal_slice = None

    # Compute the expected (voxel) measurements from the provided dimensions
    # NB: Actual measurements will differ slightly due to spine curvature
    measurements = {
        # NB: 'sct_analyze_lesion' treats lesions as cylinders. So:
        #   - Vertical axis: Length of the cylinder
        'length [mm]': dim[1],
        'midsagittal_spinal_cord_slice': mid_sagittal_slice,
        # NB: we can compute length_midsagittal_slice and width_midsagittal_slice here from dim for the purposes of
        #  testing, but in the actual script, we need the spinal cord segmentation to compute these values based on
        #  the midsagittal slice
        'length_midsagittal_slice [mm]': dim[1],
        'width_midsagittal_slice [mm]': dim[0],
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

    return measurements


@pytest.mark.sct_testing
@pytest.mark.parametrize("dummy_lesion, rtol", [
    # Straight region of `t2.nii.gz` -> little curvature -> smaller tolerance
    ([(29, 45, 25), (3, 10, 2)], 0.001),
    ([(29, 27, 25), (1, 4, 1)], 0.001),  # NB: Example from #3633
    # Curved region of `t2.nii.gz` -> lots of curvature -> larger tolerance
    ([(29, 0, 25), (4, 15, 3)], 0.01),
    # Multiple lesions
    ([[(29, 0, 25), (4, 15, 3)],
      [(29, 45, 25), (3, 10, 2)]], 0.01)
], indirect=["dummy_lesion"])
def test_sct_analyze_lesion_matches_expected_dummy_lesion_measurements(dummy_lesion, rtol, tmp_path):
    """Run the CLI script and verify that the lesion measurements match
    expected values."""
    # Run the analysis on the dummy lesion file
    path_lesion, lesion_params = dummy_lesion
    path_seg = sct_test_path("t2", "t2_seg-manual.nii.gz")
    sct_analyze_lesion.main(argv=['-m', path_lesion,
                                  '-s', path_seg,
                                  '-ofolder', str(tmp_path),
                                  '-qc', str(tmp_path / "qc")])

    # Load analysis results from pickled pandas.Dataframe
    _, fname, _ = extract_fname(path_lesion)
    with open(tmp_path/f"{fname}_analysis.pkl", 'rb') as f:
        measurements = pickle.load(f)['measures']

    # Compute expected measurements from the lesion dimensions
    for idx, (starting_coord, dim) in enumerate(lesion_params):
        expected_measurements = compute_expected_measurements(dim, starting_coord, path_seg)

        # Validate analysis results
        for key, expected_value in expected_measurements.items():
            # These measures are the same regardless of angle adjustment/spine curvature
            if key in ['volume [mm3]', 'max_axial_damage_ratio []', 'midsagittal_spinal_cord_slice']:
                np.testing.assert_equal(measurements.at[idx, key], expected_value)
            else:
                # However, these measures won't match exactly due to angle adjustment
                # from spinal cord centerline curvature
                np.testing.assert_allclose(measurements.at[idx, key],
                                           expected_value, rtol=rtol)
                # The values will be adjusted according to the cos of the angle
                # between the spinal cord centerline and the S-I axis, as per:
                # https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3681#discussion_r804822552
                # 'width' matches tissue bridge widths and 'width_midsagittal_slice [mm]'
                if key == 'max_equivalent_diameter [mm]' or 'width' in key:
                    assert measurements.at[idx, key] <= expected_value
                # 'length' matches 'length [mm]' and 'length_midsagittal_slice [mm]'
                elif 'length' in key:
                    assert measurements.at[idx, key] >= expected_value


@pytest.mark.sct_testing
@pytest.mark.parametrize("dummy_lesion, rtol", [
    # Straight region of `t2.nii.gz` -> little curvature -> smaller tolerance
    ([(29, 45, 25), (3, 10, 2)], 0.001),
    ([(29, 27, 25), (1, 4, 1)], 0.001),  # NB: Example from #3633
    # Curved region of `t2.nii.gz` -> lots of curvature -> larger tolerance
    ([(29, 0, 25), (4, 15, 3)], 0.01),
    # Multiple lesions
    ([[(29, 0, 25), (4, 15, 3)],
      [(29, 45, 25), (3, 10, 2)]], 0.01)
], indirect=["dummy_lesion"])
def test_sct_analyze_lesion_matches_expected_dummy_lesion_measurements_without_segmentation(dummy_lesion, rtol,
                                                                                            tmp_path):
    """Run the CLI script without providing SC segmentation -- only volume is computed. Max_equivalent_diameter and
    length are nan."""
    # Run the analysis on the dummy lesion file
    path_lesion, lesion_params = dummy_lesion
    sct_analyze_lesion.main(argv=['-m', path_lesion,
                                  '-ofolder', str(tmp_path),
                                  '-qc', str(tmp_path / 'qc')])  # A warning will be printed because no SC seg

    # Load analysis results from pickled pandas.Dataframe
    _, fname, _ = extract_fname(path_lesion)
    with open(tmp_path/f"{fname}_analysis.pkl", 'rb') as f:
        measurements = pickle.load(f)['measures']

    # Compute expected measurements from the lesion dimensions
    for idx, (_, dim) in enumerate(lesion_params):
        expected_measurements = compute_expected_measurements(dim)

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
def test_sct_analyze_lesion_with_template(dummy_lesion, tmp_path):
    # prep the template for use with `-f` argument of sct_analyze_lesion
    sct_register_to_template.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                                        '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                        '-l', sct_test_path('t2', 'labels.nii.gz'),
                                        '-t', sct_test_path('template'),
                                        '-ofolder', str(tmp_path)])
    sct_warp_template.main(argv=['-d', sct_test_path('t2', 't2.nii.gz'),
                                 '-w', str(tmp_path/'warp_template2anat.nii.gz'),
                                 '-a', '0',  # -a is '1' by default, but small template doesn't have atlas
                                 '-t', sct_test_path('template'),
                                 '-ofolder', str(tmp_path)])
    template_path = tmp_path / 'template'
    shutil.copy(template_path / "PAM50_small_levels.nii.gz",
                template_path / "PAM50_levels.nii.gz")  # Rename to comply with sct_analyze_lesion expectations
    (tmp_path / 'atlas').mkdir()  # make a dummy atlas folder to avoid errors due to expected folder

    # Run the analysis on the dummy lesion file
    path_lesion, _ = dummy_lesion
    sct_analyze_lesion.main(argv=['-m', path_lesion,
                                  '-f', str(tmp_path),
                                  '-ofolder', str(tmp_path)])
    assert os.path.isfile(tmp_path / "lesion_analysis.xls")
