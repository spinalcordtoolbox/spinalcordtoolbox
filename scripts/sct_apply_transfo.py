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
import sct_utils as sct

# DEFAULT PARAMETERS
class param:
    def __init__(self):
        self.debug = 1
        self.verbose = 1  # verbose
        self.dim = 3
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
    dim = param.dim

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_src = path_sct_data+'/template/MNI-Poly-AMU_T2.nii.gz'
        fname_warp_list = path_sct_data+'/t2/warp_template2anat.nii.gz'
        fname_dest = path_sct_data+'/t2/t2.nii.gz'
        verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:d:o:p:v:w:x:')
    except getopt.GetoptError:
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
        elif opt in ('-p'):
            param.interp = arg
        elif opt in ('-v'):
            verbose = int(arg)
        elif opt in ('-w'):
            fname_warp_list = arg
        elif opt in ('-x'):
            dim = arg

    # display usage if a mandatory argument is not provided
    if fname_src == '' or fname_warp_list == '' or fname_dest == '':
        usage()

    # get the right interpolation field depending on method
    interp = sct.get_interpolation('sct_antsApplyTransforms', param.interp)

    # Parse list of warping fields
    sct.printv('\nParse list of warping fields...', verbose)
    fname_warp_list = fname_warp_list.replace(' ', '')  # remove spaces
    fname_warp_list = fname_warp_list.split(",")  # parse with comma
    for i in range(len(fname_warp_list)):
        sct.printv('  Warp #'+str(i)+': '+fname_warp_list[i], verbose)

    # Check file existence
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_dest)
    for i in range(len(fname_warp_list)):
        sct.check_file_exist(fname_warp_list[i])

    # Extract path, file and extension
    path_src, file_src, ext_src = sct.extract_fname(os.path.abspath(fname_src))

    # Get output folder and file name
    if fname_src_reg == '':
        path_out = ''  # output in user's current directory
        file_out = file_src+'_reg'
        ext_out = ext_src
        fname_src_reg = path_out+file_out+ext_out
    else:
        path_out, file_out, ext_out = sct.extract_fname(fname_src_reg)

    # Apply transformation
    sct.printv('\nApply transformation...', verbose)
    # N.B. Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
    fname_warp_list.reverse()
    sct.run('sct_antsApplyTransforms -d ' + str(dim) + ' -i '+fname_src+' -o '+fname_src_reg+' -t '+' '.join(fname_warp_list)+' -r '+fname_dest+interp, verbose)

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(fname_src_reg, path_out+file_out+ext_out)

    # to view results
    print '\nDone! To view results, type:'
    print 'fslview '+fname_dest+' '+fname_src_reg+' &'
    print ''


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Apply transformations. This function is a wrapper for sct_WarpImageMultiTransform (ANTs).

USAGE
  """+os.path.basename(__file__)+""" -i <source> -d <dest> -w <warp_list>

MANDATORY ARGUMENTS
  -i <source>           source image (moving)
  -d <dest>             destination image (fixed)
  -w <warp_list>        warping field. If more than one, separate with ","

OPTIONAL ARGUMENTS
  -o <source_reg>       registered source. Default=source_reg
  -p {nn,linear,spline}  interpolation method. Default="""+str(param.interp)+"""
  -v {0,1}              verbose. Default="""+str(param.verbose)+"""
  -x {2,3}              dimension of the data. Default="""+str(param.dim)+"""
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
    param = param()
    # call main function
    main()
