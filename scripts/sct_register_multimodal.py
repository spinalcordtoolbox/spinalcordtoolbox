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

# TODO!!!!: account for non-axial destination --> reorient image
# TODO: testing script for all cases
# TODO: try to combine seg and image based for 2nd stage
# TODO: output name file for warp using "src" and "dest" file name, i.e. warp_filesrc2filedest.nii.gz
# TODO: set gradient-step-length in mm instead of vox size.

# Note for the developer: DO NOT use --collapse-output-transforms 1, otherise inverse warping field is not output


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.remove_temp_files = 1 # remove temporary files
        self.outSuffix  = "_reg"
        self.padding = 3 # add 'padding' slices at the top and bottom of the volumes if deformation at the edge is not good. Default=5. Put 0 for no padding.
#        self.convertDeformation  = 0 # Convert deformation field to 4D volume (readable by fslview)
        self.algo = 'SyN'
        self.numberIterations = "15x3" # number of iterations
        self.numberIterationsStep2 = "10" # number of iterations at step 2
        self.verbose  = 0 # verbose
        self.compute_dest2sr = 0 # compute dest2src warping field
        self.gradientStep = '0.5'  # gradientStep in SyN transformation. First value is for image-based, second is for segmentation-based (if exist)
        self.interp = 'spline'  # nn, linear, spline

import sys
import getopt
import os
import commands
import time
import sct_utils as sct

# MAIN
# ==========================================================================================
def main():

    # Initialization
    fname_src = ''
    fname_dest = ''
    fname_src_seg = ''
    fname_dest_seg = ''
    fname_output = ''
    padding = param.padding
    numberIterations = param.numberIterations
    numberIterationsStep2 = param.numberIterationsStep2
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    use_segmentation = 0 # use spinal cord segmentation to improve robustness
    # fname_init_transfo = ''
    # fname_init_transfo_inv = ''
    use_init_transfo = ''
    #gradientStep = param.gradientStep
    compute_dest2src = param.compute_dest2sr
    algo = param.algo
    start_time = time.time()
    # restrict_deformation = '1x1x1'
    print ''

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_dest = path_sct_data+'/mt/mt1.nii.gz'
        fname_src = path_sct_data+'/t2/t2.nii.gz'
        fname_dest_seg = path_sct_data+'/mt/mt1_seg.nii.gz'
        fname_src_seg = path_sct_data+'/t2/t2_seg.nii.gz'
        numberIterations = '3x0'
        numberIterationsStep2 = '1'
        gradientStep_input = '0.5'
        remove_temp_files = 0
        compute_dest2src = 1
        verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hd:g:i:m:n:o:q:r:s:t:v:x:y:z:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-d"):
            fname_dest = arg
        elif opt in ('-g'):
            gradientStep_input = arg
        elif opt in ("-i"):
            fname_src = arg
        elif opt in ("-m"):
            fname_mask = arg
        elif opt in ("-n"):
            numberIterations = arg
        elif opt in ("-o"):
            fname_output = arg
        # elif opt in ('-p'):
        #     padding = arg
        # elif opt in ('-q'):
        #     fname_init_transfo = arg
        elif opt in ('-r'):
            remove_temp_files = int(arg)
        elif opt in ("-s"):
            fname_src_seg = arg
        elif opt in ("-t"):
            fname_dest_seg = arg
        elif opt in ('-v'):
            verbose = int(arg)
        elif opt in ('-x'):
            param.interp = arg
        elif opt in ('-y'):
            numberIterationsStep2 = arg
        elif opt in ('-z'):
            compute_dest2src = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_src == '' or fname_dest == '':
        sct.printv('ERROR: Input file missing. Exit program.', 1, 'error')
        usage()

    # check segmentation data
    if (fname_src_seg != '' and fname_dest_seg == '') or (fname_src_seg == '' and fname_dest_seg != ''):
        sct.printv("ERROR: You need to select a segmentation file for the source AND the destination image. Exit program.", 1, 'error')
        usage()
    elif fname_src_seg != '' and fname_dest_seg != '':
        use_segmentation = 1

    # Check if gradient step is float
    print '\nParse gradient step...'
    try:
        float(gradientStep_input)
        gradientStep = gradientStep_input
        # gradientStep.append(gradientStep_input[i])
    except:
        sct.printv('  WARNING: Not a float. Using default value: ', 1, 'warning')
        gradientStep = param.gradientStep

    # print arguments
    print '\nInput parameters:'
    print '  Source .............. '+fname_src
    print '  Destination ......... '+fname_dest
    print '  Segmentation source . '+fname_src_seg
    print '  Segmentation dest ... '+fname_dest_seg
    # print '  Init transfo ........ '+fname_init_transfo
    print '  Output name ......... '+fname_output
    print '  Algorithm ........... '+algo
    print '  Iterations at step1 (seg) .... '+str(numberIterations)
    print '  Iterations at step2 (image) .. '+str(numberIterationsStep2)
    print '  Gradient step ....... '+gradientStep
    print '  Remove temp files ... '+str(remove_temp_files)
    print '  Verbose ............. '+str(verbose)

    # check existence of input files
    print '\nCheck if files exist...'
    sct.check_file_exist(fname_src)
    sct.check_file_exist(fname_dest)
    if use_segmentation:
        sct.check_file_exist(fname_src_seg)
        sct.check_file_exist(fname_dest_seg)

    # get full path
    fname_src = os.path.abspath(fname_src)
    fname_dest = os.path.abspath(fname_dest)
    fname_src_seg = os.path.abspath(fname_src_seg)
    fname_dest_seg = os.path.abspath(fname_dest_seg)
    # if not fname_init_transfo == '':
    #     fname_init_transfo = os.path.abspath(fname_init_transfo)  # test if not empty, otherwise it will transform the empty string into a string with path, which is a problem because the emptiness of the string is tested later.
    # if not fname_init_transfo_inv == '':
    #     fname_init_transfo_inv = os.path.abspath(fname_init_transfo_inv)
    #fname_output = os.path.abspath(fname_output)

    # Extract path, file and extension
    path_src, file_src, ext_src = sct.extract_fname(fname_src)
    path_dest, file_dest, ext_dest = sct.extract_fname(fname_dest)
    if use_segmentation:
        path_src_seg, file_src_seg, ext_src_seg = sct.extract_fname(fname_src_seg)
        path_dest_seg, file_dest_seg, ext_dest_seg = sct.extract_fname(fname_dest_seg)

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
    # file_src_tmp = 'src'
    # file_dest_tmp = 'dest'
    sct.run('sct_c3d '+fname_src+' -o '+path_tmp+'/src.nii')
    sct.run('sct_c3d '+fname_dest+' -o '+path_tmp+'/dest.nii')
    if use_segmentation:
        # file_src_seg_tmp = 'src_seg'
        # file_dest_seg_tmp = 'dest_seg'
        sct.run('sct_c3d '+fname_src_seg+' -o '+path_tmp+'/src_seg.nii.gz')
        sct.run('sct_c3d '+fname_dest_seg+' -o '+path_tmp+'/dest_seg.nii.gz')

    # go to tmp folder
    os.chdir(path_tmp)

    # Put source into destination space using header
    sct.printv('\nPut source into destination space using header...', verbose)
    sct.run('sct_antsRegistration -d 3 -t Translation[0] -m MI[dest.nii,src.nii,1,16] -c 0 -f 1 -s 0 -o [regAffine,src_regAffine.nii] -n BSpline[3]')
    if use_segmentation:
        sct.run('sct_antsRegistration -d 3 -t Translation[0] -m MI[dest_seg.nii.gz,src_seg.nii.gz,1,16] -c 0 -f 1 -s 0 -o [regAffine,src_seg_regAffine.nii.gz] -n NearestNeighbor')

    # TODO: REORIENT DATA INSTEAD! AND DO NOT FORGET TO OUTPUT AFFINE MATRIX AND CONCATENATE IT
    # # Find orientation of destination data
    # sct.printv('\nFind orientation of destination data...', verbose)
    # orientation = sct.get_orientation('dest.nii')
    # sct.printv('  '+orientation, verbose)
    # # Find the dimension corresponding to z
    # sct.printv('\nFind the dimension corresponding to the superior-inferior direction...', verbose)
    # dimension_si = 0
    # if orientation.find('I') != '-1':
    #     dimension_si = orientation.find('I')
    # elif orientation.find('S') != '-1':
    #     dimension_si = orientation.find('S')
    # else:
    #     sct.printv('ERROR in '+os.path.basename(__file__)+': Cannot find proper dimension. Exit program.', 1, 'error')
    #     sys.exit(2)
    # sct.printv('  '+str(dimension_si), verbose)
    # # Adjust ANTs variable so that the deformation is restricted in the slice plane
    # if dimension_si == 0:
    #     restrict_deformation = '0x1x1'
    # elif dimension_si == 1:
    #     restrict_deformation = '1x0x1'
    # elif dimension_si == 2:
    #     restrict_deformation = '1x1x0'
    # else:
    #     sct.printv('ERROR in '+os.path.basename(__file__)+': Cannot adjust variable: restrict_deformation. There is something wrong with your NIFTI header. Exit program.', 1, 'error')
    #     sys.exit(2)

    # # if use initial transformation (!! needs to be inserted before the --transform field in sct_antsRegistration)
    # if fname_init_transfo != '':
    #     file_src_reg_tmp = file_src_tmp+'_reg'
    #     # apply initial transformation to moving image, and then estimate transformation between this output and
    #     # destination image. This approach was chosen instead of inputting the transfo into ANTs, because if the transfo
    #     # does not bring the image to the same space as the destination image, then warping fields cannot be concatenated at the end.
    #     print('\nApply initial transformation to moving image...')
    #     sct.run('sct_apply_transfo -i '+file_src_tmp+'.nii -o '+file_src_reg_tmp+'.nii -d '+file_dest_tmp+'.nii -w '+fname_init_transfo+' -p spline')
    #     file_src_tmp = file_src_reg_tmp
    #     if use_segmentation:
    #         file_src_seg_reg_tmp = file_src_seg_tmp+'_reg'
    #         sct.run('sct_apply_transfo -i '+file_src_seg_tmp+'.nii -o '+file_src_seg_reg_tmp+'.nii -d '+file_dest_seg_tmp+'.nii -w '+fname_init_transfo+' -p spline')
    #         file_src_seg_tmp = file_src_seg_reg_tmp

    # Pad the target and source image (because ants doesn't deform the extremities)
    sct.printv('\nPad the target and source image (because ants doesn''t deform the extremities)...', verbose)
    # pad_image('src.nii', 'src_pad.nii', padding)
    pad_image('dest.nii', 'dest_pad.nii', padding)

    # don't use spinal cord segmentation
    if use_segmentation == 0:

        # Estimate transformation using ANTS
        print('\nEstimate transformation using ANTS (might take a couple of minutes)...')

        cmd = 'sct_antsRegistration \
--dimensionality 3 \
'+use_init_transfo+' \
--transform '+algo+'['+str(gradientStep[0])+',3,0] \
--metric MI['+file_dest_tmp+'.nii,'+file_src_tmp+'.nii,1,32] \
--convergence '+numberIterations+' \
--shrink-factors 2x1 \
--smoothing-sigmas 0x0mm \
--Restrict-Deformation '+restrict_deformation+' \
--output [reg,'+file_src_tmp+'_reg.nii] \
--collapse-output-transforms 1 \
--interpolation BSpline[3] \
--winsorize-image-intensities [0.005,0.995]'

        status, output = sct.run(cmd)
        if verbose:
            print output

    # use spinal cord segmentation
    elif use_segmentation == 1:

        # First, put anat in metric space

#sct_antsRegistration -d 3 -t Translation[0] -m MI[dest_seg.nii,src_seg.nii,1,16] -c 0 -f 1 -s 0 -o [regSeg0,regSeg0.nii.gz] -n NearestNeighbor
# sct_antsSliceRegularizedRegistration -m MeanSquares[dest_seg.nii,regSeg0.nii.gz,1,4,0.2] -t Translation[0.5] -i 5 -s 5 -p 5 -f 1 -o [regSeg1,regSeg1.nii.gz]
        # Estimate transformation using ANTS
        sct.printv('\nStep #1: Estimate large-scale deformation using segmentations...', verbose)

        cmd = ('sct_antsSliceRegularizedRegistration '
               '-t Translation[0.5] '
               '-m MeanSquares[dest_seg.nii.gz,src_seg_regAffine.nii.gz,1,4,0.2] '
               '-p 5 '
               '-i 5 '
               '-f 1 '
               '-s 5 '
               '-o [stage1,regSeg.nii]')
        status, output = sct.run(cmd)
        # cmd = ('sct_antsRegistration '
        #        '--dimensionality 3 '
        #        '--transform '+algo+'['+str(gradientStep[1])+',3,0] '
        #        '--metric MI['+file_dest_seg_tmp+'.nii,'+file_src_seg_tmp+'.nii,1,32] '
        #        '--convergence '+numberIterations+' '
        #        '--shrink-factors 4x1 '
        #        '--smoothing-sigmas 1x1mm '
        #        '--Restrict-Deformation '+restrict_deformation+' '
        #        '--output [regSeg,regSeg.nii]')
        # if verbose:
        #     print output

        # 2nd stage registration
        sct.printv('\nStep #2: Estimate small-scale deformations using images...', verbose)
        cmd = ('sct_antsRegistration '
               '--dimensionality 3 '
               '--initial-moving-transform stage1Warp.nii.gz '
               '--transform '+algo+'['+gradientStep+',3,0] '
               '--metric MI[dest_pad.nii,src_regAffine.nii,1,32] '
               '--convergence '+numberIterationsStep2+' '
               '--shrink-factors 1 '
               '--smoothing-sigmas 0mm '
               # '--Restrict-Deformation '+restrict_deformation+' '
               '--output [stage2,src_regAffineWarp.nii] '
               '--collapse-output-transforms 0 '
               '--interpolation BSpline[3]')
        status, output = sct.run(cmd)
        # if verbose:
        #     print output

        # Concatenate multi-stage transformations
        sct.printv('\nConcatenate multi-stage transformations...', verbose)
        sct.run('sct_concat_transfo -w stage1Warp.nii.gz,stage21Warp.nii.gz -d dest.nii -o warp_src2dest0.nii.gz')
        sct.run('sct_concat_transfo -w stage21InverseWarp.nii.gz,stage1InverseWarp.nii.gz -d src.nii -o warp_dest2src0.nii.gz')

    # Concatenate transformations
    sct.printv('\nConcatenate affine and local transformations...', verbose)
    sct.run('sct_concat_transfo -w regAffine0GenericAffine.mat,warp_src2dest0.nii.gz -d dest.nii -o warp_src2destFinal.nii.gz')
    sct.run('sct_concat_transfo -w warp_dest2src0.nii.gz,-regAffine0GenericAffine.mat -d src.nii -o warp_dest2srcFinal.nii.gz')

    # # if user has initial transfo:
    # if fname_init_transfo != '':
    #     if use_segmentation == 0:
    #         # src --> dest
    #         cmd1 = 'sct_ComposeMultiTransform 3 warp_src2dest.nii.gz -R dest.nii reg0Warp.nii.gz '+fname_init_transfo
    #         # dest --> src
    #         if compute_dest2src:
    #             cmd2 = 'sct_ComposeMultiTransform 3 warp_dest2src.nii.gz -R src.nii '+fname_init_transfo_inv+' reg0InverseWarp.nii.gz'
    #
    #     elif use_segmentation == 1:
    #         # src --> dest
    #         cmd1 = 'sct_ComposeMultiTransform 3 warp_src2dest.nii.gz -R dest.nii reg1Warp.nii.gz regSeg0Warp.nii.gz '+fname_init_transfo
    #         # dest --> src
    #         if compute_dest2src:
    #             cmd2 = 'sct_ComposeMultiTransform 3 warp_dest2src.nii.gz -R src.nii '+fname_init_transfo_inv+' regSeg0InverseWarp.nii.gz reg1InverseWarp.nii.gz'

    # # if user does not have initial transfo:
    # if use_segmentation == 0:
    #     # src --> dest
    #     cmd1 = 'sct_ComposeMultiTransform 3 warp_src2dest.nii.gz -R dest.nii reg0Warp.nii.gz'
    #     # dest --> src
    #     if compute_dest2src:
    #         cmd2 = 'sct_ComposeMultiTransform 3 warp_dest2src.nii.gz -R src.nii reg0InverseWarp.nii.gz'
    #
    # elif use_segmentation == 1:
    #     # src --> dest
    #     cmd1 = 'sct_ComposeMultiTransform 3 warp_src2dest.nii.gz -R dest.nii reg1Warp.nii.gz regSeg0Warp.nii.gz'
    #     # dest --> src
    #     if compute_dest2src:
    #         cmd2 = 'sct_ComposeMultiTransform 3 warp_dest2src.nii.gz -R src.nii regSeg0InverseWarp.nii.gz reg1InverseWarp.nii.gz'

    # print('>> ' + cmd1)
    # commands.getstatusoutput(cmd1)  # here cannot use sct.run() because of wrong output status in sct_ComposeMultiTransform
    # if compute_dest2src:
    #     print('>> ' + cmd2)
    #     commands.getstatusoutput(cmd2)  # here cannot use sct.run() because of wrong output status in sct_ComposeMultiTransform

    # Apply warping field to src data
    sct.printv('\nApply transfo source --> dest...', verbose)
    sct.run('sct_apply_transfo -i src.nii -o src_reg.nii -d dest.nii -w warp_src2destFinal.nii.gz -p '+param.interp)
    sct.printv('\nApply transfo dest --> source...', verbose)
    sct.run('sct_apply_transfo -i dest.nii -o dest_reg.nii -d src.nii -w warp_dest2srcFinal.nii.gz -p '+param.interp)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    fname_src2dest = sct.generate_output_file(path_tmp+'/src_reg.nii', path_out+file_out+ext_out)
    sct.generate_output_file(path_tmp+'/warp_src2destFinal.nii.gz', path_out+'warp_src2dest.nii.gz')
    fname_dest2src = sct.generate_output_file(path_tmp+'/dest_reg.nii', path_out+file_dest+'_reg'+ext_dest)
    sct.generate_output_file(path_tmp+'/warp_dest2srcFinal.nii.gz', path_out+'warp_dest2src.nii.gz')

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nRemove temporary files...'
        sct.run('rm -rf '+path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

    # to view results
    print '\nTo view results, type:'
    print 'fslview '+fname_dest+' '+fname_src2dest+' &'
    print 'fslview '+fname_src+' '+fname_dest2src+' &'
    print ''


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  This program co-registers two spinal cord volumes. The deformation is non-rigid and is constrained
  in the Z direction (i.e., axial plane). Hence, this function assumes that orientation of the DEST
  image is axial. If you need to register two volumes with large deformations and/or different
  contrasts, it is recommended to input spinal cord segmentations (binary mask) in order to achieve
  maximum robustness. To do so, you can use sct_segmentation_propagation.
  The program outputs a warping field that can be used to register other images to the destination
  image. To apply the warping field to another image, use sct_apply_transfo

USAGE
  """+os.path.basename(__file__)+""" -i <source> -d <dest>

MANDATORY ARGUMENTS
  -i <source>                  source image
  -d <dest>                    destination image

OPTIONAL ARGUMENTS
  -s <source_seg>              segmentation for source image (mandatory if -t is used)
  -t <dest_seg>                segmentation for destination image (mandatory if -s is used)
  -x {0,1}                     compute inverse transformation (dest --> source)
                               CURRENTLY NOT WORKING!!!
  -o <output>                  name of output file. Default=source_reg
  -n <N1xN2>                   number of iterations for first and second stage. Default="""+param.numberIterations+"""
  -y <N>                       number of iterations at step 2 (if using segmentation).
                               Set 0 to register based on segmentation only. Default="""+param.numberIterationsStep2+"""
  -g <gradientStep>            gradientStep for SyN transformation. The larger the more deformation.
                               Default="""+param.gradientStep+"""
  -x {nn,linear,spline}  Final Interpolation. Default="""+str(param.interp)+"""
  -r {0,1}                     remove temporary files. Default='+str(param.remove_temp_files)+'
  -v {0,1}                     verbose. Default="""+str(param.verbose)+"""

EXAMPLES
  1. Register mean DWI data to the T1 volume using segmentations:
    """+os.path.basename(__file__)+""" -i dwi_mean.nii.gz -d t1.nii.gz -s dwi_mean_seg.nii.gz -t t1_seg.nii.gz

  2. Register another volume to the template using previously-estimated transformations:
    """+os.path.basename(__file__)+""" -i $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -d t1.nii.gz -s $SCT_DIR/data/template/MNI-Poly-AMU_cord.nii.gz -t segmentation_binary.nii.gz -q ../t2/warp_template2anat.nii.gz -x 1 -z ../t2/warp_anat2template.nii.gz \n"""

    # exit program
    sys.exit(2)


# pad an image
# ==========================================================================================
def pad_image(fname_in, file_out, padding):
    cmd = 'sct_c3d '+fname_in+' -pad 0x0x'+str(padding)+'vox 0x0x'+str(padding)+'vox 0 -o '+file_out
    print(">> "+cmd)
    os.system(cmd)
    return
#
#
# # remove padding
# # ==========================================================================================
# def remove_padding(file_ref, file_in, file_out):
#     # remove padding by reslicing padded data into unpadded space
#     cmd = 'sct_c3d '+file_ref+' '+file_in+' -reslice-identity -o '+file_out
#     print(">> "+cmd)
#     os.system(cmd)
#     return


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
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
