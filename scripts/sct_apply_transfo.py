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
import time
from msct_parser import Parser
import sct_utils as sct
from sct_crop_image import ImageCropper


class Param:
    def __init__(self):
        self.verbose = '1'
        self.remove_tmp_files = '1'


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)
    parser.usage.set_description('Apply transformations. This function is a wrapper for antsApplyTransforms (ANTs).')
    parser.add_option(name="-i",
                      type_value="file",
                      description="input image",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-d",
                      type_value="file",
                      description="destination image",
                      mandatory=True,
                      example="out.nii.gz")
    parser.add_option(name="-w",
                      type_value=[[','], "file"],
                      description="warping field",
                      mandatory=True,
                      example="warp1.nii.gz,warp2.nii.gz")
    parser.add_option(name="-crop",
                      type_value="multiple_choice",
                      description="Crop Reference. 0 : no reference. 1 : sets background to 0. 2 : use normal background",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1', '2'])
    parser.add_option(name="-c",
                      type_value=None,
                      description="Crop Reference. 0 : no reference. 1 : sets background to 0. 2 : use normal background",
                      mandatory=False,
                      deprecated_by='-crop')
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="registered source.",
                      mandatory=False,
                      default_value='',
                      example="dest.nii.gz")
    parser.add_option(name="-x",
                      type_value="multiple_choice",
                      description="interpolation method",
                      mandatory=False,
                      default_value='spline',
                      example=['nn', 'linear', 'spline'])
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])

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
        for i in range(len(fname_warp_list)):
            # Check if inverse matrix is specified with '-' at the beginning of file name
            if fname_warp_list[i].find('-') == 0:
                use_inverse.append('-i ')
                fname_warp_list[i] = fname_warp_list[i][1:]  # remove '-'
            else:
                use_inverse.append('')
            sct.printv('  Transfo #' + str(i) + ': ' + use_inverse[i] + fname_warp_list[i], verbose)
            fname_warp_list_invert.append(use_inverse[i] + fname_warp_list[i])

        # need to check if last warping field is an affine transfo
        isLastAffine = False
        path_fname, file_fname, ext_fname = sct.extract_fname(fname_warp_list_invert[-1])
        if ext_fname in ['.txt', '.mat']:
            isLastAffine = True

        # check if destination file is 3d
        if not sct.check_if_3d(fname_dest):
            sct.printv('ERROR: Destination data must be 3d')

        # N.B. Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
        fname_warp_list_invert.reverse()

        # Extract path, file and extension
        path_src, file_src, ext_src = sct.extract_fname(fname_src)
        path_dest, file_dest, ext_dest = sct.extract_fname(fname_dest)

        # Get output folder and file name
        if fname_out == '':
            path_out = ''  # output in user's current directory
            file_out = file_src + '_reg'
            ext_out = ext_src
            fname_out = path_out + file_out + ext_out

        # Get dimensions of data
        sct.printv('\nGet dimensions of data...', verbose)
        from msct_image import Image
        nx, ny, nz, nt, px, py, pz, pt = Image(fname_src).dim
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
            sct.run('isct_antsApplyTransforms -d '+dim+' -i ' + fname_src + ' -o ' + fname_out + ' -t ' + ' '.join(fname_warp_list_invert) + ' -r ' + fname_dest + interp, verbose)

        # if 4d, loop across the T dimension
        else:
            # create temporary folder
            sct.printv('\nCreate temporary folder...', verbose)
            path_tmp = sct.slash_at_the_end('tmp.' + time.strftime("%y%m%d%H%M%S"), 1)
            # sct.run('mkdir '+path_tmp, verbose)
            sct.run('mkdir ' + path_tmp, verbose)

            # convert to nifti into temp folder
            sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
            from sct_convert import convert
            convert(fname_src, path_tmp + 'data.nii', squeeze_data=False)
            sct.run('cp ' + fname_dest + ' ' + path_tmp + file_dest + ext_dest)
            fname_warp_list_tmp = []
            for fname_warp in fname_warp_list:
                path_warp, file_warp, ext_warp = sct.extract_fname(fname_warp)
                sct.run('cp ' + fname_warp + ' ' + path_tmp + file_warp + ext_warp)
                fname_warp_list_tmp.append(file_warp + ext_warp)
            fname_warp_list_invert_tmp = fname_warp_list_tmp[::-1]

            os.chdir(path_tmp)
            # split along T dimension
            sct.printv('\nSplit along T dimension...', verbose)
            from sct_image import split_data
            im_dat = Image('data.nii')
            im_header = im_dat.hdr
            data_split_list = split_data(im_dat, 3)
            for im in data_split_list:
                im.save()

            # apply transfo
            sct.printv('\nApply transformation to each 3D volume...', verbose)
            for it in range(nt):
                file_data_split = 'data_T' + str(it).zfill(4) + '.nii'
                file_data_split_reg = 'data_reg_T' + str(it).zfill(4) + '.nii'
                status, output = sct.run('isct_antsApplyTransforms -d 3 -i ' + file_data_split + ' -o ' + file_data_split_reg + ' -t ' + ' '.join(fname_warp_list_invert_tmp) + ' -r ' + file_dest + ext_dest + interp, verbose)

            # Merge files back
            sct.printv('\nMerge file back...', verbose)
            from sct_image import concat_data
            import glob
            path_out, name_out, ext_out = sct.extract_fname(fname_out)
            # im_list = [Image(file_name) for file_name in glob.glob('data_reg_T*.nii')]
            # concat_data use to take a list of image in input, now takes a list of file names to open the files one by one (see issue #715)
            fname_list = glob.glob('data_reg_T*.nii')
            im_out = concat_data(fname_list, 3, im_header['pixdim'])
            im_out.setFileName(name_out + ext_out)
            im_out.save(squeeze_data=False)

            os.chdir('..')
            sct.generate_output_file(path_tmp + name_out + ext_out, fname_out)
            # Delete temporary folder if specified
            if int(remove_temp_files):
                sct.printv('\nRemove temporary files...', verbose)
                sct.run('rm -rf ' + path_tmp, verbose, error_exit='warning')

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

        # display elapsed time
        sct.printv('\nDone! To view results, type:', verbose)
        sct.printv('fslview ' + fname_dest + ' ' + fname_out + ' &\n', verbose, 'info')


# MAIN
# ==========================================================================================
def main(args=None):

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    input_filename = arguments["-i"]
    fname_dest = arguments["-d"]
    warp_filename = arguments["-w"]

    transform = Transform(input_filename=input_filename, fname_dest=fname_dest, warp=warp_filename)

    if "-crop" in arguments:
        transform.crop = arguments["-crop"]
    if "-o" in arguments:
        transform.output_filename = arguments["-o"]
    if "-x" in arguments:
        transform.interp = arguments["-x"]
    if "-r" in arguments:
        transform.remove_temp_files = int(arguments["-r"])
    if "-v" in arguments:
        transform.verbose = int(arguments["-v"])

    transform.apply()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # # initialize parameters
    param = Param()
    # call main function
    main()
