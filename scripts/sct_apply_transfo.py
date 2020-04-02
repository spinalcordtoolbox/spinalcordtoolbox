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

from __future__ import division, absolute_import

import sys, io, os, time, functools
import argparse

from spinalcordtoolbox.utils import Metavar, SmartFormatter
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.cropping import ImageCropper
from spinalcordtoolbox.math import dilate

import sct_utils as sct
import sct_image


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation

    parser = argparse.ArgumentParser(
        description='Apply transformations. This function is a wrapper for antsApplyTransforms (ANTs).',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py")
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
        "-v",
        help="Verbose: 0: nothing, 1: classic, 2: expended.",
        required=False,
        type=int,
        default=1,
        choices=(0, 1, 2))

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

        interp = sct.get_interpolation('isct_antsApplyTransforms', self.interp)

        # Parse list of warping fields
        sct.printv('\nParse list of warping fields...', verbose)
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
                raise ValueError("Displacement field in {} is invalid: should be encoded" \
                 " in a 5D file with vector intent code" \
                 " (see https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h" \
                 .format(path_warp))
        # need to check if last warping field is an affine transfo
        isLastAffine = False
        path_fname, file_fname, ext_fname = sct.extract_fname(fname_warp_list_invert[-1][-1])
        if ext_fname in ['.txt', '.mat']:
            isLastAffine = True

        ## check if destination file is 3d
        # sct.check_dim(fname_dest, dim_lst=[3]) # PR 2598: we decided to skip this line.

        # N.B. Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
        fname_warp_list_invert.reverse()
        fname_warp_list_invert = functools.reduce(lambda x, y: x + y, fname_warp_list_invert)

        # Extract path, file and extension
        path_src, file_src, ext_src = sct.extract_fname(fname_src)
        path_dest, file_dest, ext_dest = sct.extract_fname(fname_dest)

        # Get output folder and file name
        if fname_out == '':
            path_out = ''  # output in user's current directory
            file_out = file_src + '_reg'
            ext_out = ext_src
            fname_out = os.path.join(path_out, file_out + ext_out)

        # Get dimensions of data
        sct.printv('\nGet dimensions of data...', verbose)
        img_src = Image(fname_src)
        nx, ny, nz, nt, px, py, pz, pt = img_src.dim
        # nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_src)
        sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt), verbose)

        # if 3d
        if nt == 1:
            # Apply transformation
            sct.printv('\nApply transformation...', verbose)
            if nz in [0, 1]:
                dim = '2'
            else:
                dim = '3'
            # if labels, dilate before resampling
            if islabel:
                sct.printv("\nDilate labels before warping...")
                path_tmp = sct.tmp_create(basename="apply_transfo", verbose=verbose)
                fname_dilated_labels = os.path.join(path_tmp, "dilated_data.nii")
                # dilate points
                dilate(Image(fname_src), 2, 'ball').save(fname_dilated_labels)
                fname_src = fname_dilated_labels

            sct.printv("\nApply transformation and resample to destination space...", verbose)
            sct.run(['isct_antsApplyTransforms',
                     '-d', dim,
                     '-i', fname_src,
                     '-o', fname_out,
                     '-t'
                     ] + fname_warp_list_invert + ['-r', fname_dest] + interp, verbose=verbose, is_sct_binary=True)

        # if 4d, loop across the T dimension
        else:
            if islabel:
                raise NotImplementedError

            dim = '4'
            path_tmp = sct.tmp_create(basename="apply_transfo", verbose=verbose)

            # convert to nifti into temp folder
            sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
            img_src.save(os.path.join(path_tmp, "data.nii"))
            sct.copy(fname_dest, os.path.join(path_tmp, file_dest + ext_dest))
            fname_warp_list_tmp = []
            for fname_warp in list_warp:
                path_warp, file_warp, ext_warp = sct.extract_fname(fname_warp)
                sct.copy(fname_warp, os.path.join(path_tmp, file_warp + ext_warp))
                fname_warp_list_tmp.append(file_warp + ext_warp)
            fname_warp_list_invert_tmp = fname_warp_list_tmp[::-1]

            curdir = os.getcwd()
            os.chdir(path_tmp)

            # split along T dimension
            sct.printv('\nSplit along T dimension...', verbose)

            im_dat = Image('data.nii')
            im_header = im_dat.hdr
            data_split_list = sct_image.split_data(im_dat, 3)
            for im in data_split_list:
                im.save()

            # apply transfo
            sct.printv('\nApply transformation to each 3D volume...', verbose)
            for it in range(nt):
                file_data_split = 'data_T' + str(it).zfill(4) + '.nii'
                file_data_split_reg = 'data_reg_T' + str(it).zfill(4) + '.nii'

                status, output = sct.run(['isct_antsApplyTransforms',
                                          '-d', '3',
                                          '-i', file_data_split,
                                          '-o', file_data_split_reg,
                                          '-t',
                                          ] + fname_warp_list_invert_tmp + [
                    '-r', file_dest + ext_dest,
                ] + interp, verbose, is_sct_binary=True)

            # Merge files back
            sct.printv('\nMerge file back...', verbose)
            import glob
            path_out, name_out, ext_out = sct.extract_fname(fname_out)
            # im_list = [Image(file_name) for file_name in glob.glob('data_reg_T*.nii')]
            # concat_data use to take a list of image in input, now takes a list of file names to open the files one by one (see issue #715)
            fname_list = glob.glob('data_reg_T*.nii')
            fname_list.sort()
            im_out = sct_image.concat_data(fname_list, 3, im_header['pixdim'])
            im_out.save(name_out + ext_out)

            os.chdir(curdir)
            sct.generate_output_file(os.path.join(path_tmp, name_out + ext_out), fname_out)
            # Delete temporary folder if specified
            if remove_temp_files:
                sct.printv('\nRemove temporary files...', verbose)
                sct.rmtree(path_tmp, verbose=verbose)

        # Copy affine matrix from destination space to make sure qform/sform are the same
        sct.printv("Copy affine matrix from destination space to make sure qform/sform are the same.", verbose)
        im_src_reg = Image(fname_out)
        im_src_reg.copy_qform_from_ref(Image(fname_dest))
        im_src_reg.save(verbose=0)  # set verbose=0 to avoid warning message about rewriting file

        if islabel:
            sct.printv("\nTake the center of mass of each registered dilated labels...")
            sct.run(['sct_label_utils',
                     '-i', fname_out,
                     '-o', fname_out,
                     '-cubic-to-point'])
            if remove_temp_files:
                sct.printv('\nRemove temporary files...', verbose)
                sct.rmtree(path_tmp, verbose=verbose)

        # 2. crop the resulting image using dimensions from the warping field
        warping_field = fname_warp_list_invert[-1]
        # if last warping field is an affine transfo, we need to compute the space of the concatenate warping field:
        if isLastAffine:
            sct.printv('WARNING: the resulting image could have wrong apparent results. You should use an affine transformation as last transformation...', verbose, 'warning')
        else:
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
                    img_out = cropper.crop(background=0)
                elif crop_reference == 2:
                    img_out = cropper.crop()
                img_out.save(fname_out)

        sct.display_viewer_syntax([fname_dest, fname_out], verbose=verbose)


# MAIN
# ==========================================================================================
def main(args=None):
    """
    Entry point for sct_apply_transfo
    :param args: list of input arguments. For parameters -w and -winv, args list should include a nested list for every
    item. Example: args=['-i', 'file.nii', '-w', ['warp1.nii', 'warp2.nii']]
    :return:
    """

    # get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    else:
        # flatten the list of input arguments because -w and -winv carry a nested list
        lst = []
        for line in args:
            lst.append(line) if isinstance(line, str) else lst.extend(line)
        args = lst
    parser = get_parser()
    arguments = parser.parse_args(args=args)

    input_filename = arguments.i
    fname_dest = arguments.d
    warp_filename = arguments.w
    warpinv_filename = arguments.winv

    transform = Transform(input_filename=input_filename, fname_dest=fname_dest, list_warp=warp_filename,
                          list_warpinv=warpinv_filename)

    transform.crop = arguments.crop
    transform.output_filename = arguments.o
    transform.interp = arguments.x
    transform.remove_temp_files = arguments.r
    transform.verbose = arguments.v
    sct.init_sct(log_level=transform.verbose, update=True)  # Update log level

    transform.apply()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
