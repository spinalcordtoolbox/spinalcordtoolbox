#!/usr/bin/env python
#########################################################################################
# Register a volume (e.g., EPI from fMRI or DTI scan) to an anatomical image.
#
# See Usage() below for more information.
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# none
#
# EXTERNAL SOFTWARE
# - itksnap/sct_c3d <http://www.itksnap.org/pmwiki/pmwiki.php?n=Main.HomePage>
# - ants <http://stnava.github.io/ANTs/>
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-06-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: output name file for warp using "src" and "dest" file name, i.e. warp_filesrc2filedest.nii.gz
# TODO: testing script for all cases

# Note for the developer: DO NOT use --collapse-output-transforms 1, otherwise inverse warping field is not output

# TODO: make three possibilities:
# - one-step registration, using only image registration (by sliceReg or antsRegistration)
# - two-step registration, using first segmentation-based registration (based on sliceReg or antsRegistration) and second the image registration (and allow the choice of algo, metric, etc.)
# - two-step registration, using only segmentation-based registration


import sys
#import getopt
import shutil
import os
import commands
import time
import sct_utils as sct
from msct_parser import Parser


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.outSuffix  = "_reg"


# Parameters for registration
class Paramreg:
    def __init__(self):
        self.algo = 'syn'
        self.metric = 'MI'
        self.iter = '10'
        self.shrink = '2'
        self.smooth = '0'
        self.poly = '3'  # slicereg only
        self.gradStep = '0.5'
    # update constructor with user's parameters
    def update(self, paramreg_user):
        list_objects = paramreg_user.split(',')
        for object in list_objects:
            obj = object.split('=')
            setattr(self, obj[0], obj[1])


# MAIN
# ==========================================================================================
def main():

    # Initialization
    # fname_src = ''
    # fname_dest = ''
    fname_output = ''
    fname_mask = ''
    padding = 5
    parameters = 'algo=syn,metric=MI,iter=10,shrink=2,smooth=0,grad=0.5'
    remove_temp_files = 1
    verbose = 1

    start_time = time.time()
    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # get default parameters
    paramreg = Paramreg()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('This program co-registers two 3D volumes. The deformation is non-rigid and is '
                                 'constrained along Z direction (i.e., axial plane). Hence, this function assumes '
                                 'that orientation of the destination image is axial (RPI). If you need to register '
                                 'two volumes with large deformations and/or different contrasts, it is recommended to '
                                 'input spinal cord segmentations (binary mask) in order to achieve maximum robustness.'
                                 ' The program outputs a warping field that can be used to register other images to the'
                                 ' destination image. To apply the warping field to another image, use '
                                 'sct_apply_transfo')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Source image.",
                      mandatory=True,
                      example="data_src.nii.gz")
    parser.add_option(name="-d",
                      type_value="file",
                      description="Destination image.",
                      mandatory=True,
                      example="data_dest.nii.gz")
    parser.add_option(name="-m",
                      type_value="file",
                      description="Binary mask to improve robustness.",
                      mandatory=False,
                      example="mask.nii.gz")
    parser.add_option(name="-o",
                      type_value="str",
                      description="Name of output file.",
                      mandatory=False,
                      example="src_reg.nii.gz")
    parser.add_option(name="-p",
                      type_value="str",
                      description="""parameters for registration. Separate with ",".\nalgo: {syn,bsplinesyn,slicereg}. Default="""+paramreg.algo+"""\nmetric: {CC,MI,MeanSquares}. Default="""+paramreg.metric+"""\niter: <int> Number of iterations. Default="""+paramreg.iter+"""\nshrink: <int> Shrink factor (only for SyN). Default="""+paramreg.shrink+"""\nsmooth: <int> Smooth factor (only for SyN). Default="""+paramreg.smooth+"""\ngradStep: <float> Gradient step (only for SyN). Default="""+paramreg.gradStep+"""\npoly: <int> Polynomial degree (only for slicereg). Default="""+paramreg.poly,
                      mandatory=False,
                      default_value="algo=syn,metric=MI,iter=5,shrink=2,smooth=0,grad=0.5",
                      example="algo=slicereg,metric=MeanSquares,iter=20")
    parser.add_option(name="-z",
                      type_value="int",
                      description="""size of z-padding to enable deformation at edges when using SyN.""",
                      mandatory=False,
                      example="5")
    parser.add_option(name="-x",
                      type_value="multiple_choice",
                      description="""Final interpolation.""",
                      mandatory=False,
                      default_value='linear',
                      example=['nn', 'linear', 'spline'])
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose.""",
                      mandatory=False,
                      example=['0', '1'])
    arguments = parser.parse(sys.argv[1:])

    # get arguments
    fname_src = arguments['-i']
    fname_dest = arguments['-d']
    if '-o' in arguments:
        fname_output = arguments['-o']
    if '-m' in arguments:
        fname_mask = arguments['-m']
    if '-z' in arguments:
        padding = arguments['-z']
    if '-m' in arguments:
        fname_mask = arguments['-m']
    paramreg_user = arguments['-p']
    if '-r' in arguments:
        remove_temp_files = arguments['-r']
    interp = arguments['-x']
    if '-v' in arguments:
        verbose = arguments['-v']

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_dest = path_sct_data+'/mt/mt1.nii.gz'
        fname_src = path_sct_data+'/t2/t2.nii.gz'
        param_user = '10,syn,0.5,MI'
        remove_temp_files = 0
        verbose = 1

    # update paramreg with user's arguments
    paramreg.update(paramreg_user)

    # if sliceReg is used, we can't pad in the image...
    if paramreg.algo == 'slicereg':
        sct.printv('WARNING: if sliceReg is used, padding should not be used. Now setting padding=0', 1, 'warning')
        padding = 0

    # print arguments
    print '\nInput parameters:'
    print '  Source .............. '+fname_src
    print '  Destination ......... '+fname_dest
    print '  Mask ................ '+fname_mask
    print '  Output name ......... '+fname_output
    print '  Algorithm ........... '+paramreg.algo
    print '  Number of iterations  '+paramreg.iter
    print '  Metric .............. '+paramreg.metric
    print '  Remove temp files ... '+str(remove_temp_files)
    print '  Verbose ............. '+str(verbose)

    # # check existence of input files
    # print '\nCheck if files exist...'
    # sct.check_file_exist(fname_src)
    # sct.check_file_exist(fname_dest)
    # if not fname_mask == '':
    #     sct.check_file_exist(fname_mask)

    # Get if input is 3D
    sct.printv('\nCheck if input data are 3D...', verbose)
    sct.check_if_3d(fname_src)
    sct.check_if_3d(fname_dest)

    # check if destination data is RPI
    sct.printv('\nCheck if destination data is RPI...', verbose)
    sct.check_if_rpi(fname_dest)

    # set metricSize
    if paramreg.metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius

    # Extract path, file and extension
    path_src, file_src, ext_src = sct.extract_fname(fname_src)
    path_dest, file_dest, ext_dest = sct.extract_fname(fname_dest)

    # define output folder and file name
    if fname_output == '':
        path_out = ''  # output in user's current directory
        file_out = file_src+"_reg"
        ext_out = ext_src
    else:
        path_out, file_out, ext_out = sct.extract_fname(fname_output)

    # create temporary folder
    print('\nCreate temporary folder...')
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    status, output = sct.run('mkdir '+path_tmp)

    # copy files to temporary folder
    print('\nCopy files...')
    sct.run('sct_c3d '+fname_src+' -o '+path_tmp+'/src.nii')
    sct.run('sct_c3d '+fname_dest+' -o '+path_tmp+'/dest.nii')
    if not fname_mask == '':
        sct.run('sct_c3d '+fname_mask+' -o '+path_tmp+'/mask.nii.gz')
        masking = '-x mask.nii.gz'  # this variable will be used when calling ants
    else:
        masking = ''  # this variable will be used when calling ants

    # go to tmp folder
    os.chdir(path_tmp)

    # Pad the destination image (because ants doesn't deform the extremities)
    sct.printv('\nPad src and destination volumes (because ants doesn''t deform the extremities)...', verbose)
    pad_image('dest.nii', 'dest_pad.nii', padding)

    # Put source into destination space using header (no estimation -- purely based on header)
    sct.printv('\nPut source into destination space using header...', verbose)
    sct.run('sct_antsRegistration -d 3 -t Translation[0] -m MI[dest_pad.nii,src.nii,1,16] -c 0 -f 1 -s 0 -o [regAffine,src_regAffine.nii] -n BSpline[3]')

    # Estimate transformation
    sct.printv('\nEstimate transformation (can take a couple of minutes)...', verbose)

    if paramreg.algo == 'slicereg':
        cmd = ('sct_antsSliceRegularizedRegistration '
               '-t Translation[0.5] '
               '-m '+paramreg.metric+'[dest_pad.nii,src_regAffine.nii,1,'+metricSize+',Regular,0.2] '
               '-p '+paramreg.poly+' '
               '-i '+paramreg.iter+' '
               '-f 1 '
               '-s 0 '
               '-o [stage10,src_regAffineWarp.nii] '  # here the warp name is stage10 because antsSliceReg add "Warp"
               +masking)
    elif paramreg.algo == 'syn' or paramreg.algo == 'bsplinesyn':
        cmd = ('sct_antsRegistration '
               '--dimensionality 3 '
               '--transform '+paramreg.algo+'['+paramreg.gradStep+',3,0] '
               '--metric '+paramreg.metric+'[dest_pad.nii,src_regAffine.nii,1,'+metricSize+'] '
               '--convergence '+paramreg.iter+' '
               '--shrink-factors '+paramreg.shrink+' '
               '--smoothing-sigmas '+paramreg.smooth+'mm '
               '--restrict-deformation 1x1x0 '
               '--output [stage1,src_regAffineWarp.nii] '
               '--interpolation BSpline[3] '
               +masking)
    else:
        sct.printv('ERROR: algo '+paramreg.algo+' does not exist. Exit program', 1, 'error')

    # run registration
    status, output = sct.run(cmd)
    if status:
        sct.printv(output, 1, 'error')
        sct.printv('\nERROR: ANTs failed. Exit program.\n', 1, 'error')

    # Concatenate transformations
    sct.printv('\nConcatenate affine and local transformations...', verbose)
    sct.run('sct_concat_transfo -w regAffine0GenericAffine.mat,stage10Warp.nii.gz -d dest.nii -o warp_src2dest.nii.gz')
    sct.run('sct_concat_transfo -w stage10InverseWarp.nii.gz,-regAffine0GenericAffine.mat -d src.nii -o warp_dest2src.nii.gz')

    # Apply warping field to src data
    sct.printv('\nApply transfo source --> dest...', verbose)
    sct.run('sct_apply_transfo -i src.nii -o src_reg.nii -d dest.nii -w warp_src2dest.nii.gz -x '+interp)
    sct.printv('\nApply transfo dest --> source...', verbose)
    sct.run('sct_apply_transfo -i dest.nii -o dest_reg.nii -d src.nii -w warp_dest2src.nii.gz -x '+interp)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    fname_src2dest = sct.generate_output_file(path_tmp+'/src_reg.nii', path_out+file_out+ext_out)
    sct.generate_output_file(path_tmp+'/warp_src2dest.nii.gz', path_out+'warp_'+file_src+'2'+file_dest+'.nii.gz')
    fname_dest2src = sct.generate_output_file(path_tmp+'/dest_reg.nii', path_out+file_dest+'_reg'+ext_dest)
    sct.generate_output_file(path_tmp+'/warp_dest2src.nii.gz', path_out+'warp_'+file_dest+'2'+file_src+'.nii.gz')
    # sct.generate_output_file(path_tmp+'/warp_dest2src.nii.gz', path_out+'warp_dest2src.nii.gz')

    # Delete temporary files
    if remove_temp_files == 1:
        sct.printv('\nRemove temporary files...', verbose)
        shutil.rmtree(path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s', verbose)
    sct.printv('\nTo view results, type:', verbose)
    sct.printv('fslview '+fname_dest+' '+fname_src2dest+' &', verbose, 'info')
    sct.printv('fslview '+fname_src+' '+fname_dest2src+' &\n', verbose, 'info')



# pad an image
# ==========================================================================================
def pad_image(fname_in, file_out, padding):
    sct.run('sct_c3d '+fname_in+' -pad 0x0x'+str(padding)+'vox 0x0x'+str(padding)+'vox 0 -o '+file_out, 1)
    return



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()


# Convert deformation field to 4D volume (readable by fslview)
# DONE: clean code below-- right now it does not work
#===========
#if convertDeformation:
#    print('\nConvert deformation field...')
#    cmd = 'sct_c3d -mcs tmp.regWarp.nii -oo tmp.regWarp_x.nii tmp.regWarp_y.nii tmp.regWarp_z.nii'
#    print(">> "+cmd)
#    os.system(cmd)
#    cmd = 'fslmerge -t '+path_out+'warp_comp.nii tmp.regWarp_x.nii tmp.regWarp_y.nii tmp.regWarp_z.nii'
#    print(">> "+cmd)
#    os.system(cmd)
#===========
