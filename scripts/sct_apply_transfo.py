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
import getopt
import commands
import time
from msct_parser import Parser
import sct_utils as sct
from sct_crop_image import ImageCropper


class Transform:
    def __init__(self,input_filename, warp, output_filename, source_reg='', verbose=0, crop=0, interp='spline', debug=0):
        self.input_filename = input_filename
        if isinstance(warp, str):
            self.warp_input = list([warp])
        else:
            self.warp_input = warp
        self.output_filename = output_filename
        self.interp = interp
        self.source_reg = source_reg
        self.crop = crop
        self.verbose = verbose
        self.debug = debug

    def apply(self):
        # Initialization
        fname_src = self.input_filename  # source image (moving)
        fname_warp_list = self.warp_input  # list of warping fields
        fname_dest = self.output_filename  # destination image (fix)
        fname_src_reg = self.source_reg
        verbose = self.verbose
        fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI
        crop_reference = self.crop  # if = 1, put 0 everywhere around warping field, if = 2, real crop

        # Parameters for debug mode
        if self.debug:
            print '\n*** WARNING: DEBUG MODE ON ***\n'
            # get path of the testing data
            status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
            fname_src = path_sct_data+'/template/MNI-Poly-AMU_T2.nii.gz'
            fname_warp_list = path_sct_data+'/t2/warp_template2anat.nii.gz'
            fname_dest = path_sct_data+'/t2/t2.nii.gz'
            verbose = 1

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
            sct.printv('  Transfo #'+str(i)+': '+use_inverse[i]+fname_warp_list[i], verbose)
            fname_warp_list_invert.append(use_inverse[i]+fname_warp_list[i])

        # need to check if last warping field is an affine transfo
        isLastAffine = False
        path_fname, file_fname, ext_fname = sct.extract_fname(fname_warp_list_invert[-1])
        if ext_fname in ['.txt','.mat']:
            isLastAffine = True

        # Check file existence
        sct.printv('\nCheck file existence...', verbose)
        sct.check_file_exist(fname_src)
        sct.check_file_exist(fname_dest)
        for i in range(len(fname_warp_list)):
            # check if file exist
            sct.check_file_exist(fname_warp_list[i])

        # check if destination file is 3d
        sct.check_if_3d(fname_dest)

        # N.B. Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
        fname_warp_list_invert.reverse()

        # Extract path, file and extension
        path_src, file_src, ext_src = sct.extract_fname(fname_src)

        # Get output folder and file name
        if fname_src_reg == '':
            path_out = ''  # output in user's current directory
            file_out = file_src+'_reg'
            ext_out = ext_src
            fname_out = path_out+file_out+ext_out
        else:
            fname_out = fname_src_reg

        # Get dimensions of data
        sct.printv('\nGet dimensions of data...', verbose)
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_src)
        sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)+ ' x ' + str(nt), verbose)

        # if 3d
        if nt == 1:
            # Apply transformation
            sct.printv('\nApply transformation...', verbose)
            sct.run('isct_antsApplyTransforms -d 3 -i '+fname_src+' -o '+fname_out+' -t '+' '.join(fname_warp_list_invert)+' -r '+fname_dest+interp, verbose)

        # if 4d, loop across the T dimension
        else:
            # create temporary folder
            sct.printv('\nCreate temporary folder...', verbose)
            path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
            # sct.run('mkdir '+path_tmp, verbose)
            sct.run('mkdir '+path_tmp, verbose)

            # Copying input data to tmp folder
            # NB: cannot use c3d here because c3d cannot convert 4D data.
            sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
            sct.run('cp '+fname_src+' '+path_tmp+'data'+ext_src, verbose)
            # go to tmp folder
            os.chdir(path_tmp)
            try:
                # convert to nii format
                sct.run('fslchfiletype NIFTI data', verbose)

                # split along T dimension
                sct.printv('\nSplit along T dimension...', verbose)
                sct.run(fsloutput+'fslsplit data data_T', verbose)
                # apply transfo
                sct.printv('\nApply transformation to each 3D volume...', verbose)
                for it in range(nt):
                    file_data_split = 'data_T'+str(it).zfill(4)+'.nii'
                    file_data_split_reg = 'data_reg_T'+str(it).zfill(4)+'.nii'
                    sct.run('isct_antsApplyTransforms -d 3 -i '+file_data_split+' -o '+file_data_split_reg+' -t '+' '.join(fname_warp_list_invert)+' -r '+fname_dest+interp, verbose)

                # Merge files back
                sct.printv('\nMerge file back...', verbose)
                cmd = fsloutput+'fslmerge -t '+fname_out
                for it in range(nt):
                    file_data_split_reg = 'data_reg_T'+str(it).zfill(4)+'.nii'
                    cmd = cmd+' '+file_data_split_reg
                sct.run(cmd, verbose)

            except:
                pass
            # come back to parent folder
            os.chdir('..')

        # 2. crop the resulting image using dimensions from the warping field
        warping_field = fname_warp_list_invert[-1]
        # if last warping field is an affine transfo, we need to compute the space of the concatenate warping field:
        if isLastAffine:
            sct.printv('WARNING: the resulting image could have wrong apparent results. You should use an affine transformation as last transformation...',1,'warning')
        elif crop_reference == 1:
            ImageCropper(input_file=fname_out, output_file=fname_out, ref=warping_field, background=0).crop()
            # sct.run('sct_crop_image -i '+fname_out+' -o '+fname_out+' -ref '+warping_field+' -b 0')
        elif crop_reference == 2:
            ImageCropper(input_file=fname_out, output_file=fname_out, ref=warping_field).crop()
            # sct.run('sct_crop_image -i '+fname_out+' -o '+fname_out+' -ref '+warping_field)

        # display elapsed time
        sct.printv('\nDone! To view results, type:', verbose)
        sct.printv('fslview '+fname_dest+' '+fname_out+' &\n', verbose, 'info')


if __name__ == "__main__":

    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description('Apply transformations. This function is a wrapper for antsApplyTransforms (ANTs).')
    parser.add_option(name="-i",
                      type_value="image_nifti",
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
    parser.add_option(name="-c",
                      type_value="int",
                      description="Crop Reference. 0 : no reference. 1 : sets background to 0. 2 : use normal background",
                      mandatory=False,
                      default_value='0',
                      example=['0','1','2'])
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="output file",
                      mandatory=False,
                      default_value='',
                      example="source.nii.gz")
    parser.add_option(name="-x",
                      type_value="multiple_choice",
                      description="interpolation method",
                      mandatory=False,
                      default_value='spline',
                      example=['nn','linear','spline'])

    arguments = parser.parse(sys.argv[1:])

    input_filename = arguments["-i"]
    output_filename = arguments["-d"]
    warp_filename = arguments["-w"]

    transform = Transform(input_filename=input_filename, output_filename=output_filename, warp=warp_filename)

    if "-c" in arguments:
        transform.crop = arguments["-c"]
    if "-o" in arguments:
        transform.source_reg = arguments["-o"]
    if "-x" in arguments:
        transform.interp = arguments["-x"]

    transform.apply()
