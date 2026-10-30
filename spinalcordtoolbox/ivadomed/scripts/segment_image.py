#!/usr/bin/env python
"""
This script applies a trained model on a single image. Output are generated in the current directory.
"""

# TODO: create entry_points in setup.py and update docstrings usage
# TODO: 'add_suffix' and 'splitext' should be moved to utils library (if it makes sense).

import argparse
import os
import nibabel as nib

from ivadomed import inference as imed_inference
from ivadomed import utils as imed_utils


def get_parser():
    parser = argparse.ArgumentParser(
        prog='segment_image',
        description='Applies a trained model on a single image. Output are generated in the current directory.')
    parser.add_argument("-i", "--image", nargs='+', required=True,
                        help="Image(s) to segment. You can specify more than one image (separate with space).",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("-m", "--model", required=True,
                        help="Path to folder that contains ONNX and/or PT model and ivadomed JSON config file.",
                        metavar=imed_utils.Metavar.folder)
    parser.add_argument("-s", "--suffix", default="_pred",
                        help="Suffix to add to the input image. Default: '_pred'",
                        metavar=imed_utils.Metavar.str)
    return parser


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension.

    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:
    .. code:: python

        add_suffix(t2.nii, _mean) -> t2_mean.nii
        add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """
    stem, ext = splitext(fname)
    return stem + suffix + ext


def splitext(fname):
    """
    Split a fname (folder/file + ext) into a folder/file and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    dir_, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            break
    else:
        stem, ext = os.path.splitext(filename)

    return os.path.join(dir_, stem), ext


def segment_image(fname_images: str, path_model: str, suffix_out: str, options: dict):
    """
    Applies a trained model on image(s). Output predictions are generated in the current directory.

    For example::
    
        ivadomed_segment_image -i t2s.nii.gz -m /usr/bob/my_model_directory

    Args:
        fname_images (str): Image(s) to segment. You can specify more than one image (separate with space). Flag: ``--image``, ``-i``
        path_model (str): Path to folder that contains ONNX and/or PT model and ivadomed JSON config file. Flag: ``--model``, ``-m``
        suffix_out (str): Suffix to add to the input image. Default: '_pred'. Flag: ``--suffix-out``, ``-s``
        options (dict): Options to pass to `imed_inference.segment_volume`.

    Returns:
        None
    """
    nii_lst, target_lst = imed_inference.segment_volume(path_model, fname_images, options=options)

    for i in range(len(nii_lst)):
        # TODO (minor): make path_out output images in the same dir as the input image.
        path_out = './'
        file_out = add_suffix(os.path.basename(fname_images[i]), suffix_out)
        nib.save(nii_lst[i], os.path.join(path_out, file_out))

    # TODO: add to support PNG
    # imed_inference.pred_to_png(nii_lst, target_lst, "<path_to_the_source_file>/image")

    # TODO: display a nice message at the end with syntax for FSLeyes if input is a NIfTI file.


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)
    # options = {"pixel_size": [0.13, 0.13], "overlap_2D": [48, 48], "binarize_maxpooling": True}
    # TODO: the 'no_patch' option does not seem to work as expected, because only a fraction of the image is segmented.
    # options = {"no_patch": True}
    options = {}
    segment_image(args.image, args.model, args.suffix, options)


if __name__ == '__main__':
    main()
