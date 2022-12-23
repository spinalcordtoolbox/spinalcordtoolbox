#!/usr/bin/env python
#########################################################################################
#
# Apply transformations. This function is a wrapper for sct_WarpImageMultiTransform
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Olivier Comtois
# Modified: 2014-07-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: display message at the end
# TODO: interpolation methods

import sys
import os
import functools
from typing import Sequence

from spinalcordtoolbox.image import Image, generate_output_file
from spinalcordtoolbox.cropping import ImageCropper
from spinalcordtoolbox.math import dilate
from spinalcordtoolbox.labels import cubic_to_point
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, get_interpolation, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, run_proc, printv, set_loglevel
from spinalcordtoolbox.utils.fs import tmp_create, rmtree, extract_fname, copy

from spinalcordtoolbox.scripts import sct_image


# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description='Apply transformations. This function is a wrapper for antsApplyTransforms (ANTs).'
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        "-i",
        required=True,
        help='Input image. Example: t2.nii.gz',
        metavar=Metavar.file,
    )
    mandatoryArguments.add_argument(
        "-d",
        required=True,
        help='Destination image. Example: out.nii.gz',
        metavar=Metavar.file,
    )
    mandatoryArguments.add_argument(
        "-w",
        nargs='+',
        required=True,
        help='Transformation(s), which can be warping fields (nifti image) or affine transformation matrix (text '
             'file). Separate with space. Example: warp1.nii.gz warp2.nii.gz',
        metavar=Metavar.file,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-winv",
        help='Affine transformation(s) listed in flag -w which should be inverted before being used. Note that this '
             'only concerns affine transformation (not warping fields). If you would like to use an inverse warping '
             'field, then directly input the inverse warping field in flag -w.',
        nargs='+',
        metavar=Metavar.file,
        default=[])
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-crop",
        help="Crop Reference. 0: no reference, 1: sets background to 0, 2: use normal background.",
        required=False,
        type=int,
        default=0,
        choices=(0, 1, 2))
    optional.add_argument(
        "-o",
        help='Registered source. Example: dest.nii.gz',
        required=False,
        metavar=Metavar.file,
        default='')
    optional.add_argument(
        "-x",
        help=""" Interpolation method. The 'label' method is to be used if you would like to apply a transformation
        on a file that has single-voxel labels (classical interpolation methods won't work, as resampled labels might
        disappear or their values be altered). The function will dilate each label, apply the transformation using
        nearest neighbour interpolation, and then take the center-of-mass of each "blob" and output a single voxel per
        blob.""",
        required=False,
        default='spline',
        choices=('nn', 'linear', 'spline', 'label'))
    optional.add_argument(
        "-r",
        help="""Remove temporary files.""",
        required=False,
        type=int,
        default=1,
        choices=(0, 1))
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


class Transform:
    def __init__(self, input_filename, fname_dest, list_warp, list_warpinv=[], output_filename='', verbose=0, crop=0,
                 interp='spline', remove_temp_files=1, debug=0):
        self.input_filename = input_filename
        self.list_warp = list_warp
        self.list_warpinv = list_warpinv
        self.fname_dest = fname_dest
        self.output_filename = output_filename
        self.interp = interp
        self.crop = crop
        self.verbose = verbose
        self.remove_temp_files = remove_temp_files
        self.debug = debug

    def apply(self):
        # Initialization
        fname_src = self.input_filename  # source image (moving)
        list_warp = self.list_warp  # list of warping fields
        fname_out = self.output_filename  # output
        fname_dest = self.fname_dest  # destination image (fix)
        verbose = self.verbose
        remove_temp_files = self.remove_temp_files
        crop_reference = self.crop  # if = 1, put 0 everywhere around warping field, if = 2, real crop

        islabel = False
        if self.interp == 'label':
            islabel = True
            self.interp = 'nn'

        interp = get_interpolation('isct_antsApplyTransforms', self.interp)

        # Parse list of warping fields
        printv('\nParse list of warping fields...', verbose)
        use_inverse = []
        fname_warp_list_invert = []
        # list_warp = list_warp.replace(' ', '')  # remove spaces
        # list_warp = list_warp.split(",")  # parse with comma
        for idx_warp, path_warp in enumerate(self.list_warp):
            # Check if this transformation should be inverted
            if path_warp in self.list_warpinv:
                use_inverse.append('-i')
                # list_warp[idx_warp] = path_warp[1:]  # remove '-'
                fname_warp_list_invert += [[use_inverse[idx_warp], list_warp[idx_warp]]]
            else:
                use_inverse.append('')
                fname_warp_list_invert += [[path_warp]]
            path_warp = list_warp[idx_warp]
            if path_warp.endswith((".nii", ".nii.gz")) \
                    and Image(list_warp[idx_warp]).header.get_intent()[0] != 'vector':
                raise ValueError("Displacement field in {} is invalid: should be encoded"
                                 " in a 5D file with vector intent code"
                                 " (see https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h"
                                 .format(path_warp))
        # need to check if last warping field is an affine transfo
        isLastAffine = False
        path_fname, file_fname, ext_fname = extract_fname(fname_warp_list_invert[-1][-1])
        if ext_fname in ['.txt', '.mat']:
            isLastAffine = True

        # check if destination file is 3d
        # check_dim(fname_dest, dim_lst=[3]) # PR 2598: we decided to skip this line.

        # N.B. Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
        fname_warp_list_invert.reverse()
        fname_warp_list_invert = functools.reduce(lambda x, y: x + y, fname_warp_list_invert)

        # Extract path, file and extension
        path_src, file_src, ext_src = extract_fname(fname_src)
        path_dest, file_dest, ext_dest = extract_fname(fname_dest)

        # Get output folder and file name
        if fname_out == '':
            path_out = ''  # output in user's current directory
            file_out = file_src + '_reg'
            ext_out = ext_src
            fname_out = os.path.join(path_out, file_out + ext_out)

        # Get dimensions of data
        printv('\nGet dimensions of data...', verbose)
        img_src = Image(fname_src)
        nx, ny, nz, nt, px, py, pz, pt = img_src.dim
        # nx, ny, nz, nt, px, py, pz, pt = get_dimension(fname_src)
        printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt), verbose)

        # if 3d
        if nt == 1:
            # Apply transformation
            printv('\nApply transformation...', verbose)
            if nz in [0, 1]:
                dim = '2'
            else:
                dim = '3'
            # if labels, dilate before resampling
            if islabel:
                printv("\nDilate labels before warping...")
                path_tmp = tmp_create(basename="apply-transfo-3d-label")
                fname_dilated_labels = os.path.join(path_tmp, "dilated_data.nii")
                # dilate points
                dilate(Image(fname_src), 4, 'ball').save(fname_dilated_labels)
                fname_src = fname_dilated_labels

            printv("\nApply transformation and resample to destination space...", verbose)
            run_proc(['isct_antsApplyTransforms',
                      '-d', dim,
                      '-i', fname_src,
                      '-o', fname_out,
                      '-t'
                      ] + fname_warp_list_invert + ['-r', fname_dest] + interp, is_sct_binary=True)

        # if 4d, loop across the T dimension
        else:
            if islabel:
                raise NotImplementedError

            dim = '4'
            path_tmp = tmp_create(basename="apply-transfo-4d")

            # convert to nifti into temp folder
            printv('\nCopying input data to tmp folder and convert to nii...', verbose)
            img_src.save(os.path.join(path_tmp, "data.nii"))
            copy(fname_dest, os.path.join(path_tmp, file_dest + ext_dest))
            fname_warp_list_tmp = []
            for fname_warp in list_warp:
                path_warp, file_warp, ext_warp = extract_fname(fname_warp)
                copy(fname_warp, os.path.join(path_tmp, file_warp + ext_warp))
                fname_warp_list_tmp.append(file_warp + ext_warp)
            fname_warp_list_invert_tmp = fname_warp_list_tmp[::-1]

            curdir = os.getcwd()
            os.chdir(path_tmp)

            # split along T dimension
            printv('\nSplit along T dimension...', verbose)

            im_dat = Image('data.nii')
            im_header = im_dat.hdr
            data_split_list = sct_image.split_data(im_dat, 3)
            for im in data_split_list:
                im.save()

            # apply transfo
            printv('\nApply transformation to each 3D volume...', verbose)
            for it in range(nt):
                file_data_split = 'data_T' + str(it).zfill(4) + '.nii'
                file_data_split_reg = 'data_reg_T' + str(it).zfill(4) + '.nii'

                status, output = run_proc(['isct_antsApplyTransforms',
                                           '-d', '3',
                                           '-i', file_data_split,
                                           '-o', file_data_split_reg,
                                           '-t',
                                           ] + fname_warp_list_invert_tmp + [
                    '-r', file_dest + ext_dest,
                ] + interp, verbose, is_sct_binary=True)

            # Merge files back
            printv('\nMerge file back...', verbose)
            import glob
            path_out, name_out, ext_out = extract_fname(fname_out)
            # im_list = [Image(file_name) for file_name in glob.glob('data_reg_T*.nii')]
            # concat_data use to take a list of image in input, now takes a list of file names to open the files one by one (see issue #715)
            fname_list = glob.glob('data_reg_T*.nii')
            fname_list.sort()
            im_list = [Image(fname) for fname in fname_list]
            im_out = sct_image.concat_data(im_list, 3, im_header['pixdim'])
            im_out.save(name_out + ext_out)

            os.chdir(curdir)
            generate_output_file(os.path.join(path_tmp, name_out + ext_out), fname_out)
            # Delete temporary folder if specified
            if remove_temp_files:
                printv('\nRemove temporary files...', verbose)
                rmtree(path_tmp, verbose=verbose)

        # Copy affine matrix from destination space to make sure qform/sform are the same
        printv("Copy affine matrix from destination space to make sure qform/sform are the same.", verbose)
        im_src_reg = Image(fname_out)
        im_src_reg.copy_qform_from_ref(Image(fname_dest))
        im_src_reg.save(verbose=0)  # set verbose=0 to avoid warning message about rewriting file

        if islabel:
            printv("\nTake the center of mass of each registered dilated labels...")
            labeled_img = cubic_to_point(im_src_reg)
            labeled_img.save(path=fname_out)
            if remove_temp_files:
                printv('\nRemove temporary files...', verbose)
                rmtree(path_tmp, verbose=verbose)

        # Crop the resulting image using dimensions from the warping field
        warping_field = fname_warp_list_invert[-1]
        # If the last transformation is not an affine transfo, we need to compute the matrix space of the concatenated
        # warping field
        if not isLastAffine and crop_reference in [1, 2]:
            printv('Last transformation is not affine.')
            if crop_reference in [1, 2]:
                # Extract only the first ndim of the warping field
                img_warp = Image(warping_field)
                if dim == '2':
                    img_warp_ndim = Image(img_src.data[:, :], hdr=img_warp.hdr)
                elif dim in ['3', '4']:
                    img_warp_ndim = Image(img_src.data[:, :, :], hdr=img_warp.hdr)
                # Set zero to everything outside the warping field
                cropper = ImageCropper(Image(fname_out))
                cropper.get_bbox_from_ref(img_warp_ndim)
                if crop_reference == 1:
                    printv('Cropping strategy is: keep same matrix size, put 0 everywhere around warping field')
                    img_out = cropper.crop(background=0)
                elif crop_reference == 2:
                    printv('Cropping strategy is: crop around warping field (the size of warping field will '
                           'change)')
                    img_out = cropper.crop()
                img_out.save(fname_out)


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    """
    Entry point for sct_apply_transfo
    :param argv: list of input arguments.
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    input_filename = arguments.i
    fname_dest = arguments.d
    warp_filename = arguments.w
    warpinv_filename = arguments.winv

    transform = Transform(input_filename=input_filename, fname_dest=fname_dest, list_warp=warp_filename,
                          list_warpinv=warpinv_filename)

    transform.crop = arguments.crop
    transform.output_filename = fname_out = arguments.o
    transform.interp = arguments.x
    transform.remove_temp_files = arguments.r
    transform.verbose = verbose

    transform.apply()

    display_viewer_syntax([fname_dest, fname_out], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
