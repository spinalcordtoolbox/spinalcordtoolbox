#!/usr/bin/env python
#########################################################################################
#
# Apply transformations. This function is a wrapper for sct_WarpImageMultiTransform
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
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

import sct_utils as sct




# DEFAULT PARAMETERS
class Param:
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.interp = 'spline'  # nn, trilinear, spline


# main
#=======================================================================================================================
def main():

    # Initialization
    fname_src = ''  # source image (moving)
    fname_warp_list = ''  # list of warping fields
    fname_dest = ''  # destination image (fix)
    fname_src_reg = ''
    verbose = 1
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI
    crop_reference = 0 # if = 1, put 0 everywhere around warping field, if = 2, real crop

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_src = path_sct_data+'/template/MNI-Poly-AMU_T2.nii.gz'
        fname_warp_list = path_sct_data+'/t2/warp_template2anat.nii.gz'
        fname_dest = path_sct_data+'/t2/t2.nii.gz'
        verbose = 1
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:d:o:v:w:x:c:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-i'):
                fname_src = arg
            elif opt in ('-d'):
                fname_dest = arg
            elif opt in ('-o'):
                fname_src_reg = arg
            elif opt in ('-x'):
                param.interp = arg
            elif opt in ('-v'):
                verbose = int(arg)
            elif opt in ('-w'):
                fname_warp_list = arg
            elif opt in ('-c'):
                crop_reference = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_src == '' or fname_warp_list == '' or fname_dest == '':
        usage()

    # get the right interpolation field depending on method
    interp = sct.get_interpolation('isct_antsApplyTransforms', param.interp)

    # Parse list of warping fields
    sct.printv('\nParse list of warping fields...', verbose)
    use_inverse = []
    fname_warp_list_invert = []
    fname_warp_list = fname_warp_list.replace(' ', '')  # remove spaces
    fname_warp_list = fname_warp_list.split(",")  # parse with comma
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
    for i in range(len(fname_warp_list_invert)):
        sct.check_file_exist(fname_warp_list_invert[i])

    # check if destination file is 3d
    sct.check_if_3d(fname_dest)

    # N.B. Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
    fname_warp_list_invert.reverse()

    # Extract path, file and extension
    # path_src, file_src, ext_src = sct.extract_fname(os.path.abspath(fname_src))
    # fname_dest = os.path.abspath(fname_dest)
    path_src, file_src, ext_src = sct.extract_fname(fname_src)
    # fname_dest = os.path.abspath(fname_dest

    # Get output folder and file name
    if fname_src_reg == '':
        path_out = ''  # output in user's current directory
        file_out = file_src+'_reg'
        ext_out = ext_src
        fname_out = path_out+file_out+ext_out
    else:
    #     path_out, file_out, ext_out = sct.extract_fname(fname_src_reg)
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
        sct.run('mkdir '+path_tmp, verbose)

        # Copying input data to tmp folder
        # NB: cannot use c3d here because c3d cannot convert 4D data.
        sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
        sct.run('cp '+fname_src+' '+path_tmp+'data'+ext_src, verbose)
        # go to tmp folder
        os.chdir(path_tmp)
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
        sct.run(cmd, param.verbose)
        # come back to parent folder
        os.chdir('..')

    # 2. crop the resulting image using dimensions from the warping field
    warping_field = fname_warp_list_invert[-1]
    # if last warping field is an affine transfo, we need to compute the space of the concatenate warping field:
    if isLastAffine:
        sct.printv('WARNING: the resulting image could have wrong apparent results. You should use an affine transformation as last transformation...',1,'warning')
    elif crop_reference == 1:
        sct.run('sct_crop_image -i '+fname_out+' -o '+fname_out+' -ref '+warping_field+' -b 0')
    elif crop_reference == 2:
        sct.run('sct_crop_image -i '+fname_out+' -o '+fname_out+' -ref '+warping_field)

    # display elapsed time
    sct.printv('\nDone! To view results, type:', verbose)
    sct.printv('fslview '+fname_dest+' '+fname_out+' &\n', verbose, 'info')



# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Apply transformations. This function is a wrapper for antsApplyTransforms (ANTs).

USAGE
  """+os.path.basename(__file__)+""" -i <source> -d <dest> -w <warp_list>

MANDATORY ARGUMENTS
  -i <source>           source image (moving). Can be 3D or 4D.
  -d <dest>             destination image (fixed). Must be 3D.
  -w <warp_list>        warping field. If more than one, separate with ","

OPTIONAL ARGUMENTS
  -o <source_reg>       registered source. Default=source_reg
  -x {nn,linear,spline}  interpolation method. Default="""+str(param_default.interp)+"""
  -v {0,1}              verbose. Default="""+str(param_default.verbose)+"""
  -h                    help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" \n"""

    # exit program
    sys.exit(2)



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
