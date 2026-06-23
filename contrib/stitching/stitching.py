# Copyright 2026 The TPTBox Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This file is vendored from TPTBox (https://github.com/Hendrik-code/TPTBox)
# Source: TPTBox/stitching/stitching.py
#
# Citation:
#   Graf, R., Platzek, PS., Riedel, E.O. et al.
#   Generating synthetic high-resolution spinal STIR and T1w images from T2w FSE and low-resolution axial Dixon.
#   Eur Radiol (2024). https://doi.org/10.1007/s00330-024-11047-1

from __future__ import annotations

import itertools
from pathlib import Path

import nibabel as nib
import nibabel.processing as nip
import numpy as np
from nibabel.affines import apply_affine
from nibabel.nifti1 import Nifti1Image
from scipy.ndimage import binary_opening, distance_transform_edt
from scipy.spatial import ConvexHull
from skimage.exposure import match_histograms


def get_rotation_and_spacing_from_affine(affine: np.ndarray):
    # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return rotation, spacing


def get_ras_affine(rotation, spacing, origin) -> np.ndarray:
    # https://github.com/fepegar/torchio/blob/5983f83f0e7f13f9c5056e25f8753b03426ae18a/src/torchio/data/io.py#L357
    rotation_zoom = rotation * spacing
    translation_ras = rotation.dot(origin)
    affine = np.eye(4)
    affine[:3, :3] = rotation_zoom
    affine[:3, 3] = translation_ras
    return affine


def get_all_corner_points(affine, shape):
    lst = list(itertools.product([0, 1], repeat=3))
    lst = np.array(lst) * np.array(shape)
    lst += 1

    return apply_affine(affine, lst)


def get_array(nii: Nifti1Image) -> np.ndarray:
    return np.asanyarray(nii.dataobj, dtype=nii.dataobj.dtype).copy()  # type: ignore


def set_array(nii: Nifti1Image, arr: np.ndarray) -> Nifti1Image:
    if nii.dataobj.dtype == arr.dtype:  # type: ignore
        nii = Nifti1Image(arr, nii.affine, nii.header)
    else:
        nii = Nifti1Image(get_array(nii), nii.affine, nii.header)
        nii.set_data_dtype(arr.dtype)
        nii = Nifti1Image(arr, nii.affine, nii.header)
    return nii


def argmin(lst: list):
    return lst.index(min(lst))


def get_max_affine_and_shape(points: np.ndarray, affines, min_spacing=None, dtype: type = float, verbose=False):
    hull = ConvexHull(points)

    min_possible_volume = hull.volume
    min_rotation = None
    min_volume = float("inf")
    min_shape = [0, 0, 0]
    origen = [0, 0, 0]
    spacings = []
    opt_id = -1

    # print(points[hull.vertices])
    # Find best rotation
    for idx, affine in enumerate(affines, 1):
        rotation, spacing = get_rotation_and_spacing_from_affine(affine)
        spacings.append(np.abs(spacing))

        points_rotated = points.copy()
        for i in range(points.shape[0]):
            points_rotated[i] = rotation.T.dot(points[i])

        hull_np = points_rotated[hull.vertices]

        max_v = np.max(hull_np, axis=0)
        min_v = np.min(hull_np, axis=0)
        dif = max_v - min_v
        v = dif[0] * dif[1] * dif[2]

        if v <= min_volume:
            min_volume = v
            min_rotation = rotation
            min_shape = dif
            origen = (min_v, max_v)
            opt_id = idx
    if min_rotation is None:
        raise ValueError(affines)

    new_spacing = np.min(np.round(np.stack(spacings), decimals=6), 0)
    if min_spacing is not None and min_spacing != 0:
        new_spacing = np.maximum(min_spacing, new_spacing)

    shape: np.ndarray = np.ceil(min_shape / new_spacing)
    print("Choose the following spacing:", new_spacing) if verbose else None
    print(f"Output shape is {shape}, which utilizes {min_possible_volume / min_volume * 100:.1f} % of all voxels.") if verbose else None
    affine = get_ras_affine(min_rotation, new_spacing, origen[0])
    print("The new origin is ", np.round(affine[:3, 3], 2)) if verbose else None
    print("The optimal rotation came from file number ", opt_id, " ", np.round(min_rotation.reshape(-1), 2)) if verbose else None
    return nib.Nifti1Image(np.zeros(shape.astype(int), dtype=dtype), affine)  # type: ignore


def compute_crop_slice(nii: Nifti1Image, minimum=0, dist=0):
    """
    Computes the minimum slice that removes unused space from the image and returns the corresponding slice tuple along with the origin shift required for centroids.

    Args:
        minimum (int): The minimum value of the array (0 for MRI, -1024 for CT). Default value is 0.
        dist (int): The amount of padding to be added to the cropped image. Default value is 0.
        other_crop (tuple[slice,...], optional): A tuple of slice objects representing the slice of an other image to be combined with the current slice. Default value is None.

    Returns:
        ex_slice: A tuple of slice objects that need to be applied to crop the image.
        origin_shift: A tuple of integers representing the shift required to obtain the centroids of the cropped image.

    Note:
        - The computed slice removes the unused space from the image based on the minimum value.
        - The padding is added to the computed slice.
        - If the computed slice reduces the array size to zero, a ValueError is raised.
        - If other_crop is not None, the computed slice is combined with the slice of another image to obtain a common region of interest.
        - Only None slice is supported for combining slices.
    """
    shp = nii.shape
    zms = nii.header.get_zooms()  # type: ignore
    d = np.around(dist / np.asarray(zms)).astype(int)
    array = get_array(nii)  # + minimum
    msk_bin = np.zeros(array.shape, dtype=bool)
    # bool_arr[array<minimum] = 0
    msk_bin[array > minimum] = 1
    # msk_bin = np.asanyarray(bool_arr, dtype=bool)
    msk_bin[np.isnan(msk_bin)] = 0
    cor_msk = np.where(msk_bin > 0)
    if cor_msk[0].shape[0] == 0:
        raise ValueError("Array would be reduced to zero size")
    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = max(0, c_min[0] - d[0])
    y0 = max(0, c_min[1] - d[1])
    z0 = max(0, c_min[2] - d[2])
    x1 = min(shp[0], c_max[0] + d[0])
    y1 = min(shp[1], c_max[1] + d[1])
    z1 = min(shp[2], c_max[2] + d[2])
    ex_slice = (slice(x0, x1 + 1), slice(y0, y1 + 1), slice(z0, z1 + 1))
    return ex_slice


def dilate_msk(msk_i_data: np.ndarray, mm: int = 5, connectivity: int = 3):
    from scipy.ndimage import binary_dilation, generate_binary_structure

    """
    Dilates the binary segmentation mask by the specified number of voxels.

    Args:
        mm (int, optional): The number of voxels to dilate the mask by. Defaults to 5.
        connectivity (int, optional): Elements up to a squared distance of connectivity from the center are considered neighbors.
        connectivity may range from 1 (no diagonal elements are neighbors) to rank (all elements are neighbors).
        inplace (bool, optional): Whether to modify the mask in place or return a new object. Defaults to False.
        verbose (bool, optional): Whether to print a message indicating that the mask was dilated. Defaults to True.

    Returns:
        NII: The dilated mask.

    Notes:
        The method uses binary dilation with a 3D structuring element to dilate the mask by the specified number of voxels.

    """
    struct = generate_binary_structure(3, connectivity)
    out = msk_i_data.copy() * 0
    for i in np.unique(msk_i_data):
        if i == 0:
            continue
        data = msk_i_data.copy()
        data[i != data] = 0
        msk_ibe_data = binary_dilation(data, structure=struct, iterations=mm)
        out[out == 0] = msk_ibe_data[out == 0]
    return out.astype(np.uint8)


def n4_bias_field_correction(
    nib: Nifti1Image,
    mask=None,
    threshold=60,
    shrink_factor=4,
    convergence=None,
    spline_param=150,
    verbose=False,
    weight_mask=None,
    crop=False,
):
    try:
        import ants.utils.bias_correction as bc  # pip install antspyx==0.4.2
        from ants.utils.convert_nibabel import from_nibabel, to_nibabel

    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("n4 bias field correction uses ants install it with pip install antspyx==0.4.2") from err
        # NOTE (denisseantunez, issue #5106): this fallback was previously written for newer ants
        # (5.3+) versions that moved bias_correction to a different module path. SCT already bundles
        # some ANTs functionality as precompiled binaries (e.g. isct_antsRegistration). If N4 bias
        # correction is added that way in the future, this function could be rewritten to shell out
        # to a binary instead of requiring antspyx.

    if convergence is None:
        convergence = {"iters": [50, 50, 50, 50], "tol": 1e-07}
    input_ants = from_nibabel(nib)

    if threshold != 0:
        mask = get_array(nib)
        mask[mask < threshold] = 0
        mask[mask != 0] = 1
        mask = mask.astype(np.uint8)
        mask = dilate_msk(mask, mm=3)
        mask = from_nibabel(set_array(nib, mask))

    out = bc.n4_bias_field_correction(
        input_ants,
        mask=mask,
        shrink_factor=shrink_factor,
        convergence=convergence,
        spline_param=spline_param,
        verbose=verbose,
        weight_mask=weight_mask,
    )
    out_nib = to_nibabel(out)
    if crop:
        # Crop to regions that had a normalization applied. Removes a lot of dead space
        dif = to_nibabel(input_ants - out)
        da = get_array(dif)
        da[da != 0] = 1
        dif = set_array(dif, da)
        ex_slice = compute_crop_slice(dif)
        out_nib = out_nib.slicer[ex_slice]

    return out_nib


buffer_references = {}


def buffer_reference(path, bias_field: bool, crop=False):
    if path in buffer_references:
        return buffer_references[path]
    reference = n4_bias_field_correction(nib.load(path), crop) if bias_field else get_array(nib.load(path))  # type: ignore
    buffer_references[path] = reference
    return reference


type_mapping = {
    "float": float,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
}


def main(  # noqa: C901
    images: list[str] | list[Path] | list[nib.nifti1.Nifti1Image],
    output: str | None,
    match_histogram: bool = False,
    store_ramp: bool = False,
    verbose: bool = False,
    min_value: float = 0,
    crop_empty: bool = False,
    histogram: str | None = None,
    ramp_edge_min_value=5,
    min_spacing: None | int = None,
    kick_out_fully_integrated_images=False,
    is_segmentation: bool = False,
    dtype: type | str = float,
    save=True,
):
    # This is to prevent using antspyx depencies. Set to True if the dependency is ever added
    bias_field = False
    crop_to_bias_field = False
    np.set_printoptions(precision=2, floatmode="fixed")
    if is_segmentation:
        bias_field = False
        crop_to_bias_field = False
        min_value = 0
        match_histogram = False
        histogram = None
    if len(images) == 0 or len(images) == 1:
        print("!!! Need at least two images (-i ...nii.gz ...nii.gz) to stitch!!!\n Got " + str(images))
        return None, None
    corners = []
    affines = []
    niis: list[nib.nifti1.Nifti1Image] = []
    print("### loading ###") if verbose else None
    for f_name in images:
        if isinstance(f_name, (Path, str)):
            print("Load ", f_name, Path(f_name)) if verbose else None
            # Load Nii
            nii: nib.nifti1.Nifti1Image = nib.load(f_name)  # type: ignore

        else:
            nii = f_name
        if bias_field:
            nii = n4_bias_field_correction(nii, crop=crop_to_bias_field)
        # Histogram equalization.
        if match_histogram:
            if histogram is None:
                if len(niis) == 0:
                    reference = None
                else:
                    print("Histogram equalization with previous file") if verbose else None
                    reference = get_array(niis[-1])
            elif histogram.isdigit():
                print("Histogram equalization", images[int(histogram)]) if verbose else None
                reference = buffer_reference(images[int(histogram)], bias_field=bias_field, crop=crop_to_bias_field)  # type: ignore
            else:
                print("Histogram equalization with file", histogram) if verbose else None
                reference = buffer_reference(histogram, bias_field=bias_field, crop=crop_to_bias_field)  # type: ignore
            if reference is not None:
                image = get_array(nii)

                matched = match_histograms(image.astype(float), reference.astype(float))
                matched[matched <= min_value] = min_value
                nii = set_array(nii, matched)

        niis.append(nii)
        # Get affine and points for minimum enclosing Rectangle calculation
        affine = nii.affine
        affines.append(affine)

        corners_current = get_all_corner_points(affine, nii.shape)
        corners.append(corners_current)

    corners_current = np.concatenate(corners, axis=0)

    # compute output shape and affine
    print("### compute output shape and affine ###") if verbose else None
    if is_segmentation:
        max_value = max([x.get_fdata().max() for x in niis])
        if max_value < 256:
            dtype2 = np.uint8
        elif max_value < 256 * 256:
            dtype2 = np.uint16
        elif max_value < 256 * 256 * 256 * 256:
            dtype2 = np.uint32
        else:
            dtype2 = np.uint64
        dtype = dtype2
    else:
        dtype2 = float
    nii_out = get_max_affine_and_shape(corners_current, affines, min_spacing=min_spacing, dtype=dtype2, verbose=verbose)
    target_list = []
    occupancy_list = []
    # get resampled arrays and occupancy
    print("### resample to new space ###") if verbose else None
    for i, nii in enumerate(niis, 1):
        print(f"{i:2}/{len(niis):2} resampled", end="\r") if verbose else None
        nii_new = nip.resample_from_to(nii, nii_out, 0 if is_segmentation else 3, mode="constant", cval=min_value)
        arr_new = get_array(nii_new)
        target_list.append(arr_new)
        b = nib.Nifti1Image(get_array(nii) * 0 + 1, affine=nii.affine)  # type: ignore
        b = nip.resample_from_to(b, nii_new, 0, cval=0, mode="constant")
        if is_segmentation:
            x = arr_new > 0
            occupancy_list.append((get_array(b) * x.astype(np.int8)).astype(np.float32))  # Keep segmentation if other is 0

        else:
            occupancy_list.append(get_array(b).astype(np.float32))

    print("\n### ramp stitching ###") if verbose else None
    # ramp stitching
    combinations = list(itertools.combinations(range(len(target_list)), 2))
    for idx, item in enumerate(combinations, 1):
        print(f"{idx:2}/{len(combinations):2} ramp stitching", end="\r") if verbose else None
        # TODO fix intersection with more than two occupancies
        arr_1_full = occupancy_list[item[0]]
        arr_2_full = occupancy_list[item[1]]
        ###
        structure = np.ones((ramp_edge_min_value, ramp_edge_min_value, ramp_edge_min_value), dtype=bool)
        arr_1: np.ndarray = arr_1_full.copy()
        arr_2: np.ndarray = arr_2_full.copy()
        overlap = (arr_1 * arr_2) > 0.0
        if overlap.sum() > 0:
            arr_1_ = (arr_1 > 0.0).astype(np.float32) - overlap
            arr_2_ = (arr_2 > 0.0).astype(np.float32) - overlap
            if ramp_edge_min_value == 0:
                arr_1_opened: np.ndarray = arr_1_
                arr_2_opened: np.ndarray = arr_2_
            else:
                arr_1_opened: np.ndarray = binary_opening(arr_1_, structure=structure, iterations=1, brute_force=True)
                arr_2_opened: np.ndarray = binary_opening(arr_2_, structure=structure, iterations=1, brute_force=True)

            arr_1[overlap] = distance_transform_edt(1.0 - arr_2_opened)[overlap]  # type: ignore
            arr_2[overlap] = distance_transform_edt(1.0 - arr_1_opened)[overlap]  # type: ignore
            arr_1_[overlap] = arr_1[overlap]
            arr_2_[overlap] = arr_2[overlap]
            sum_ = arr_1_ + arr_2_
            sum_[sum_ == 0] = 1.0
            arr_1_full = arr_1 / sum_
            arr_2_full = arr_2 / sum_
            if arr_1_full.max() != 1:
                import warnings

                warnings.warn(
                    str((arr_1_full.min(), arr_1_full.max())) + " the image in fully incorporated insight of an other " + str(images),
                    stacklevel=4,
                )
                if kick_out_fully_integrated_images:
                    images.pop(item[0])

            elif arr_2_full.max() != 1:
                import warnings

                warnings.warn(
                    str((arr_2_full.min(), arr_2_full.max())) + " the image in fully incorporated insight of an other " + str(images),
                    stacklevel=4,
                )
                if kick_out_fully_integrated_images:
                    images.pop(item[1])
            if (arr_1_full.max() != 1 or arr_2_full.max() != 1) and kick_out_fully_integrated_images:
                print("kick_out_fully_integrated_images")

                print(images)
                return main(
                    images,
                    output,
                    match_histogram,
                    store_ramp,
                    verbose,
                    min_value,
                    bias_field,
                    crop_to_bias_field,
                    crop_empty,
                    histogram,
                    ramp_edge_min_value,
                    min_spacing,
                    kick_out_fully_integrated_images,
                    save,
                )
            # assert arr_1_full.max() == 1, (arr_1_full.min(), arr_1_full.max())
            # assert arr_2_full.max() == 1, (arr_2_full.min(), arr_2_full.max())
            occupancy_list[item[0]] = arr_1_full
            occupancy_list[item[1]] = arr_2_full
        else:
            continue
    occupancy_arr = np.stack(occupancy_list)
    if is_segmentation:
        occupancy_arr = np.round(occupancy_arr)  # TODO assuming only two intersecting regions
    target_arr = np.stack(target_list) * occupancy_arr
    if is_segmentation:
        target_arr = target_arr.astype(dtype2)
    target_arr = target_arr.sum(0)
    target_arr[target_arr <= min_value] = min_value
    print("\n### Save ###") if verbose else None
    if output is not None:
        output = str(output)
        if not output.endswith(".nii.gz"):
            output = output.replace(".nii", "") + ".nii.gz"
        if "/" not in output and "\\" not in output:
            assert isinstance(images[0], (str, Path)), "automatic path fetching only possible if images are strings or Path, not objects"
            output = str(Path(Path(images[0]).parent, output))
    dtype = type_mapping.get(dtype, dtype)  # type: ignore
    nii_out = set_array(nii_out, target_arr.astype(dtype))
    if bias_field:
        nii_out = n4_bias_field_correction(nii_out)
    if crop_empty:
        nii_occ = set_array(nii_out, occupancy_arr)
        ex_slice = compute_crop_slice(nii_occ)
        nii_out = nii_out.slicer[ex_slice]
    else:
        ex_slice = ()

    nii_out.set_data_dtype(dtype)

    if save:
        nib.save(nii_out, output)  # type: ignore
        print("Saved ", output) if verbose else None

    if store_ramp:
        occupancy_arr = np.stack(occupancy_list, -1)
        if crop_empty:
            occupancy_arr = occupancy_arr[ex_slice]
        assert output is not None
        nii_occ = set_array(nii_out, occupancy_arr)
        output = output.replace(".nii.gz", "_ramps.nii.gz")
        if save:
            nib.save(nii_occ, output)  # type: ignore
            print("Saved ", output) if verbose else None
        return nii_out, nii_occ
    print("\n### Finished ###") if verbose else None
    return nii_out, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="nii-stitching")
    parser.add_argument("-i", "--images", nargs="+", default=[], help="filenames of images")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="out.nii.gz",
        help="filename of output image",
    )
    parser.add_argument(
        "-hist_n",
        "--histogram_name",
        type=str,
        default=None,
        help="use this file for histogram_matching instead",
    )
    help_str = "fits the histogram, for the previous in the file list. "
    parser.add_argument(
        "-hists",
        "--match_histogram",
        default=False,
        action="store_true",
        help=help_str,
    )
    help_str = "n4_bias_field_correction"
    parser.add_argument(
        "-no_bias",
        "--no_bias_field_correction",
        default=False,
        action="store_true",
        help=help_str,
    )
    help_str = "crop with generated n4_bias_field_correction"
    parser.add_argument(
        "-bias_crop",
        "--bias_field_correction_crop",
        default=False,
        action="store_true",
        help=help_str,
    )
    help_str = "crop black spaces"
    parser.add_argument("-crop", "--crop", default=False, action="store_true", help=help_str)
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    help_str = "intersecting images are bended together by there distance from vowels that are not intersecting. This flag saves the blending as a 4d nii."
    parser.add_argument("-sr", "--store_ramp", default=False, action="store_true", help=help_str)
    help_str = "If two images cut in a way, that would leave a thin slice of less than x voxels pixel, it will not be considered for the ramp calculation."
    parser.add_argument("-ramp_e", "--ramp_edge_min_value", type=int, default=5, help=help_str)
    help_str = "all values below will be set to min_value. (MRI=0, CT<=-1024)"
    parser.add_argument("-min_value", "--min_value", type=int, default=0, help=help_str)
    parser.add_argument("-ms", "--min_spacing", type=int, default=None, help="")
    parser.add_argument("-seg", "--is_segmentation", default=False, action="store_true")
    parser.add_argument("-dtype", "--dtype", default=float, type=str, help="output type")
    args = parser.parse_args()
    if args.verbose:
        try:
            from pprint import pprint

            pprint(args.__dict__)
        except Exception:
            print(args)

    main(
        args.images,
        args.output,
        args.match_histogram,
        args.store_ramp,
        args.verbose,
        bias_field=not args.no_bias_field_correction,
        crop_to_bias_field=args.bias_field_correction_crop,
        crop_empty=args.crop,
        ramp_edge_min_value=args.ramp_edge_min_value,
        histogram=args.histogram_name,
        min_value=args.min_value,
        min_spacing=args.min_spacing,
        is_segmentation=args.is_segmentation,
        dtype=args.dtype,)
