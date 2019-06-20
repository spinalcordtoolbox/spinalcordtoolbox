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
import sct_utils as sct
import sct_convert
import sct_image
import spinalcordtoolbox.image as msct_image
from sct_crop_image import ImageCropper


class Param:
    def __init__(self):
        self.verbose = '1'
        self.remove_temp_files = '1'


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation

    parser = argparse.ArgumentParser(
        description='Apply transformations. This function is a wrapper for antsApplyTransforms (ANTs).',
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatoryArguments = parser.add_argument_group("\nMandatory arguments")
    mandatoryArguments.add_argument("-i",
                        help="input image (e.g.,\"t2.nii.gz\")",
                        required = True
                        )
    mandatoryArguments.add_argument("-d",
                        help="destination image (e.g.,\"out.nii.gz\")",
                        required = True
                        )
    mandatoryArguments.add_argument("-w",
                        help="Transformation, which can be a warping field (nifti image) or an affine transformation matrix (text file). (e.g.,\"warp1.nii.gz, warp2.nii.gz\")",
                        required = True
                        )
    optional = parser.add_argument_group("\nOptional arguments")
    optional.add_argument("-h",
                          "--help",
                          action="help",
                          help="show this help message and exit"
                          )
    optional.add_argument("-crop",
                        help="Crop Reference. 0 : no reference. 1 : sets background to 0. 2 : use normal background",
                        required=False,
                        default= 0,
                        choices=(0, 1, 2))
    optional.add_argument("-c",
                        help="Crop Reference. 0 : no reference. 1 : sets background to 0. 2 : use normal background",
                        required=False,
                        )
    optional.add_argument("-o",
                        help="registered source. (e.g.,\"dest.nii.gz\")",
                        required = False,
                        default = ''
                        )
    optional.add_argument("-x",
                        help="interpolation method (e.g.,['nn', 'linear', 'spline'])",
                        required=False,
                        default='spline',
                        choices=('nn', 'linear', 'spline'))
    optional.add_argument("-r",
                        help="""Remove temporary files.""",
                        required = False,
                        default = 1,
                        choices = (0, 1))
    optional.add_argument("-v",
                        help="Verbose: 0 = nothing, 1 = classic, 2 = expended.",
                        required = False,
                        default = 1,
                        choices = (0, 1, 2))

    return parser


class Transform:
    def __init__(self, input_filename, warp, fname_dest, output_filename='', verbose=0, crop=0, interp='spline', remove_temp_files=1, debug=0):
        self.input_filename = input_filename
        if isinstance(warp, str):
            self.warp_input = list([warp])
        else:
            self.warp_input = warp
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
        fname_warp_list = self.warp_input  # list of warping fields
        fname_out = self.output_filename  # output
        fname_dest = self.fname_dest  # destination image (fix)
        verbose = self.verbose
        remove_temp_files = self.remove_temp_files
        crop_reference = self.crop  # if = 1, put 0 everywhere around warping field, if = 2, real crop

        interp = sct.get_interpolation('isct_antsApplyTransforms', self.interp)

        # Parse list of warping fields
        sct.printv('\nParse list of warping fields...', verbose)
        use_inverse = []
        fname_warp_list_invert = []
        # fname_warp_list = fname_warp_list.replace(' ', '')  # remove spaces
        # fname_warp_list = fname_warp_list.split(",")  # parse with comma
        for idx_warp, path_warp in enumerate(fname_warp_list):
            # Check if inverse matrix is specified with '-' at the beginning of file name
            if path_warp.startswith("-"):
                use_inverse.append('-i')
                fname_warp_list[idx_warp] = path_warp[1:]  # remove '-'
                fname_warp_list_invert += [[use_inverse[idx_warp], fname_warp_list[idx_warp]]]
            else:
                use_inverse.append('')
                fname_warp_list_invert += [[path_warp]]
            path_warp = fname_warp_list[idx_warp]
            if path_warp.endswith((".nii", ".nii.gz")) \
             and msct_image.Image(fname_warp_list[idx_warp]).header.get_intent()[0] != 'vector':
                raise ValueError("Displacement field in {} is invalid: should be encoded" \
                 " in a 5D file with vector intent code" \
                 " (see https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h" \
                 .format(path_warp))
        # need to check if last warping field is an affine transfo
        isLastAffine = False
        path_fname, file_fname, ext_fname = sct.extract_fname(fname_warp_list_invert[-1][-1])
        if ext_fname in ['.txt', '.mat']:
            isLastAffine = True

        # check if destination file is 3d
        if not sct.check_if_3d(fname_dest):
            sct.printv('ERROR: Destination data must be 3d')

        # N.B. Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
        fname_warp_list_invert.reverse()
        fname_warp_list_invert = functools.reduce(lambda x,y: x+y, fname_warp_list_invert)

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
        img_src = msct_image.Image(fname_src)
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
            sct.run(['isct_antsApplyTransforms',
              '-d', dim,
              '-i', fname_src,
              '-o', fname_out,
              '-t',
             ] + fname_warp_list_invert + [
             '-r', fname_dest,
             ] + interp, verbose=verbose, is_sct_binary=True)

        # if 4d, loop across the T dimension
        else:
            path_tmp = sct.tmp_create(basename="apply_transfo", verbose=verbose)

            # convert to nifti into temp folder
            sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
            img_src.save(os.path.join(path_tmp, "data.nii"))
            sct.copy(fname_dest, os.path.join(path_tmp, file_dest + ext_dest))
            fname_warp_list_tmp = []
            for fname_warp in fname_warp_list:
                path_warp, file_warp, ext_warp = sct.extract_fname(fname_warp)
                sct.copy(fname_warp, os.path.join(path_tmp, file_warp + ext_warp))
                fname_warp_list_tmp.append(file_warp + ext_warp)
            fname_warp_list_invert_tmp = fname_warp_list_tmp[::-1]

            curdir = os.getcwd()
            os.chdir(path_tmp)

            # split along T dimension
            sct.printv('\nSplit along T dimension...', verbose)

            im_dat = msct_image.Image('data.nii')
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
            if int(remove_temp_files):
                sct.printv('\nRemove temporary files...', verbose)
                sct.rmtree(path_tmp, verbose=verbose)

        # 2. crop the resulting image using dimensions from the warping field
        warping_field = fname_warp_list_invert[-1]
        # if last warping field is an affine transfo, we need to compute the space of the concatenate warping field:
        if isLastAffine:
            sct.printv('WARNING: the resulting image could have wrong apparent results. You should use an affine transformation as last transformation...', verbose, 'warning')
        elif crop_reference == 1:
            ImageCropper(input_file=fname_out, output_file=fname_out, ref=warping_field, background=0).crop()
            # sct.run('sct_crop_image -i '+fname_out+' -o '+fname_out+' -ref '+warping_field+' -b 0')
        elif crop_reference == 2:
            ImageCropper(input_file=fname_out, output_file=fname_out, ref=warping_field).crop()
            # sct.run('sct_crop_image -i '+fname_out+' -o '+fname_out+' -ref '+warping_field)

        sct.display_viewer_syntax([fname_dest, fname_out], verbose=verbose)


# MAIN
# ==========================================================================================
def main(arguments):

    input_filename = arguments.i
    fname_dest = arguments.d
    warp_filename = arguments.w

    transform = Transform(input_filename=input_filename, fname_dest=fname_dest, warp=warp_filename)

    if vars(arguments)["crop"] != None:
        transform.crop = arguments.crop
    if vars(arguments)["o"] != None:
        transform.output_filename = arguments.o
    if vars(arguments)["x"] != None:
        transform.interp = arguments.x
    if vars(arguments)["r"] != None:
        transform.remove_temp_files = arguments.r
    transform.verbose = arguments.v
    sct.init_sct(log_level=transform.verbose, update=True)  # Update log level

    transform.apply()

# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    parser = get_parser()
    arguments = parser.parse_args()
    # initialize parameters
    param = Param()
    # call main function
    main(arguments)
