import argparse
from ivadomed import utils as imed_utils
import ivadomed.preprocessing as imed_preprocessing
import nibabel as nib
import numpy as np
import ivadomed.maths as imed_maths
import ivadomed.loader.utils as imed_loader_utils
from pathlib import Path


def mask2label(path_label, aim=0):
    """Retrieve points coordinates and value from a label file containing singl voxel label.

    Args:
        path_label (str): path of nifti image
        aim (int): -1 will return all points with label between 3 and 30 , any other int > 0
            will return only the coordinates of points with label defined by aim.

    Returns:
        ndarray: array containing the asked point in the format [x,y,z,value] in the RAS orientation.

    """
    image = nib.load(path_label)
    image = nib.as_closest_canonical(image)
    arr = np.array(image.dataobj)
    list_label_image = []
    # Arr non zero used since these are single voxel label
    for i in range(len(arr.nonzero()[0])):
        x = arr.nonzero()[0][i]
        y = arr.nonzero()[1][i]
        z = arr.nonzero()[2][i]
        # need to check every points
        if aim == 0:
            # we don't want to account for pmj (label 49) nor C1/C2 which is hard to distinguish.
            if arr[x, y, z] < 30 and arr[x, y, z] != 1:
                list_label_image.append([x, y, z, arr[x, y, z]])
        elif aim > 0:
            if arr[x, y, z] == aim:
                list_label_image.append([x, y, z, arr[x, y, z]])
    list_label_image.sort(key=lambda x: x[3])
    return list_label_image


def extract_mid_slice_and_convert_coordinates_to_heatmaps(path, suffix, aim=-1):
    """
    This function takes as input a path to a dataset  and generates a set of images:
    (i) mid-sagittal image and
    (ii) heatmap of disc labels associated with the mid-sagittal image.

    Example::

        ivadomed_prepare_dataset_vertebral_labeling -p path/to/bids -s _T2w -a 0

    Args:
        path (string): path to BIDS dataset form which images will be generated.
            Flag: ``--path``, ``-p``
        suffix (string): suffix of image that will be processed (e.g., T2w).
            Flag: ``--suffix``, ``-s``
        aim (int): If aim is not 0, retrieves only labels with value = aim, else create heatmap
            with all labels. Flag: ``--aim``, ``-a``

    Returns:
        None. Images are saved in BIDS folder
    """
    t = [path_object.name for path_object in Path(path).iterdir() if path_object.name != 'derivatives']

    for i in range(len(t)):
        subject = t[i]
        path_image = Path(path, subject, 'anat', subject + suffix + '.nii.gz')
        if path_image.is_file():
            path_label = Path(path, 'derivatives', 'labels', subject, 'anat', subject + suffix +
                    '_labels-disc-manual.nii.gz')
            list_points = mask2label(str(path_label), aim=aim)
            image_ref = nib.load(path_image)
            nib_ref_can = nib.as_closest_canonical(image_ref)
            imsh = np.array(nib_ref_can.dataobj).shape
            mid_nifti = imed_preprocessing.get_midslice_average(str(path_image), list_points[0][0], slice_axis=0)
            nib.save(mid_nifti, Path(path, subject, 'anat', subject + suffix + '_mid.nii.gz'))
            lab = nib.load(path_label)
            nib_ref_can = nib.as_closest_canonical(lab)
            label_array = np.zeros(imsh[1:])

            for j in range (len(list_points)):
                label_array[list_points[j][1], list_points[j][2]] = 1

            heatmap = imed_maths.heatmap_generation(label_array[:, :], 10)
            arr_pred_ref_space = imed_loader_utils.reorient_image(np.expand_dims(heatmap[:, :], axis=0), 2, lab, nib_ref_can)
            nib_pred = nib.Nifti1Image(arr_pred_ref_space, lab.affine)
            nib.save(nib_pred, Path(path, 'derivatives', 'labels', subject, 'anat', subject + suffix +
                                            '_mid_heatmap' + str(aim) + '.nii.gz'))
        else:
            pass


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", dest="path", required=True, type=str,
                        help="Path to bids folder",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("-s", "--suffix", dest="suffix", required=True, type=str,
                        help="""Suffix of the input file as in
                                sub-xxxSUFFIX.nii.gz (E.g., _T2w)""",
                        metavar=imed_utils.Metavar.str)
    parser.add_argument("-a", "--aim", dest="aim", default=-1, type=int,
                        help="""-1 or positive int. If set to any positive int,
                                only label with this value will be taken into account""",
                        metavar=imed_utils.Metavar.int)
    return parser


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)
    extract_mid_slice_and_convert_coordinates_to_heatmaps(path=args.path, suffix=args.suffix,
                                                          aim=args.aim)


if __name__ == '__main__':
    main()
