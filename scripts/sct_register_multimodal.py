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
# - itksnap/c3d <http://www.itksnap.org/pmwiki/pmwiki.php?n=Main.HomePage>
# - ants <http://stnava.github.io/ANTs/>
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-04-05
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: testing script for all cases
# TODO: try to combine seg and image based for 2nd stage
# TODO: output name file for warp using "src" and "dest" file name, i.e. warp_filesrc2filedest.nii.gz
# TODO: flag to output warping field
# TODO: check if destination is axial orientation
# TODO: set gradient-step-length in mm instead of vox size.


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug               = 0
        self.remove_temp_files   = 1 # remove temporary files
        self.outSuffix           = "_reg"
        self.padding             = 3 # add 'padding' slices at the top and bottom of the volumes if deformation at the edge is not good. Default=5. Put 0 for no padding.
#        self.convertDeformation  = 0 # Convert deformation field to 4D volume (readable by fslview)
        self.numberIterations    = "50x30" # number of iterations
        self.verbose             = 0 # verbose


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
    gradientStepLength = '0.1' # TODO: use that?
    numberIterations = param.numberIterations
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    use_segmentation = 0 # use spinal cord segmentation to improve robustness
    fname_init_transfo = ''
    use_init_transfo = ''
    output_warping_field = "tmp.regSeg0Warp.nii.gz"
    start_time = time.time()

    # get path of the toolbox
    path_script = os.path.dirname(__file__)
    path_sct = path_script[:-8] # TODO: make it cleaner!

    # Parameters for debug mode
    if param.debug:
        # without using segmentation (minor displacement and similar contrast)
        #fname_src = os.path.expanduser("~")+'/code/spinalcordtoolbox_dev/testing/data/errsm_24/mt/mt0.nii.gz'
        #fname_dest = os.path.expanduser("~")+'/code/spinalcordtoolbox_dev/testing/data/errsm_24/mt/mt1.nii.gz'
        # using segmentation + initial transformation
        fname_src = path_sct+'/data/template/MNI-Poly-AMU_T2.nii.gz'
        fname_dest = path_sct+'/testing/data/errsm_24/mt/mt0.nii.gz'
        fname_src_seg = path_sct+'/data/template/MNI-Poly-AMU_cord.nii.gz'
        fname_dest_seg = path_sct+'/testing/data/errsm_24/mt/segmentation_binary.nii.gz'
        fname_init_transfo = path_sct+'/testing/data/errsm_24/template/warp_template2anat.nii.gz'
        numberIterations = '50x20'
        verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'he:d:i:m:n:o:p:q:r:s:t:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-d"):
            fname_dest = arg
        elif opt in ('-e'):
            extentDist = arg
        elif opt in ("-i"):
            fname_src = arg
        elif opt in ("-m"):
            fname_mask = arg
        elif opt in ("-n"):
            numberIterations = arg
        elif opt in ("-o"):
            fname_output = arg
        elif opt in ('-p'):
            padding = arg
        elif opt in ('-q'):
            fname_init_transfo = arg
        elif opt in ('-r'):
            remove_temp_files = int(arg)
        elif opt in ("-s"):
            fname_src_seg = arg
        elif opt in ("-t"):
            fname_dest_seg = arg
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_src == '' or fname_dest == '':
        usage()

    # check segmentation data
    if (fname_src_seg != '' and fname_dest_seg == '') or (fname_src_seg == '' and fname_dest_seg != ''):
        print "ERROR: You have to select a segmentation file for BOTH the source and the destination image.\nExit program."
        sys.exit(2)
    elif fname_src_seg != '' and fname_dest_seg != '':
        use_segmentation = 1

    # check existence of input files
    sct.check_file_exist(fname_src)
    sct.check_file_exist(fname_dest)
    if use_segmentation:
        sct.check_file_exist(fname_src_seg)
        sct.check_file_exist(fname_dest_seg)

    # print arguments
    print '\nCheck parameters:'
    print '.. Source:               '+fname_src
    print '.. Destination:          '+fname_dest
    print '.. Segmentation source:  '+fname_src_seg
    print '.. Segmentation dest:    '+fname_dest_seg
    print '.. Init transfo:         '+fname_init_transfo
    print '.. Output name:          '+fname_output
    #print '.. Mask:                 '+fname_mask
    print '.. number of iterations: '+str(numberIterations)
    print '.. Verbose:              '+str(verbose)
    print '.. Remove temp files:    '+str(remove_temp_files)
    #print '.. gradient step:    '+str(gradientStepLength)
    #print '.. metric type:      '+metricType

    # Extract path, file and extension
    path_src, file_src, ext_src = sct.extract_fname(fname_src)
    path_dest, file_dest, ext_dest = sct.extract_fname(fname_dest)
    if use_segmentation:
        path_src_seg, file_src_seg, ext_src_seg = sct.extract_fname(fname_src_seg)
        path_dest_seg, file_dest_seg, ext_dest_seg = sct.extract_fname(fname_dest_seg)

    # define output folder and file name
    if fname_output == '':
        path_out = path_src
        file_out = file_src+"_reg"
        ext_out = ext_src
    else:
        path_out, file_out, ext_out = sct.extract_fname(fname_output)

    # create local temp files
    print('\nCreate local temp files...')
    file_src_tmp = 'tmp.src'
    file_dest_tmp = 'tmp.dest'
    sct.run('cp '+fname_src+' '+file_src_tmp+ext_src)
    sct.run('cp '+fname_dest+' '+file_dest_tmp+ext_dest)
    if use_segmentation:
        file_src_seg_tmp = 'tmp.src_seg'
        file_dest_seg_tmp = 'tmp.dest_seg'
        sct.run('cp '+fname_src_seg+' '+file_src_seg_tmp+ext_src_seg)
        sct.run('cp '+fname_dest_seg+' '+file_dest_seg_tmp+ext_dest_seg)

    # if use initial transformation (!! needs to be inserted before the --transform field in antsRegistration)
    if fname_init_transfo != '':
        file_src_reg_tmp = file_src_tmp+'_reg'
        file_src_seg_reg_tmp = file_src_seg_tmp+'_reg'
        # apply initial transformation to moving image, and then estimate transformation between this output and
        # destination image. This approach was chosen instead of inputting the transfo into ANTs, because if the transfo
        # does not bring the image to the same space as the destination image, then warping fields cannot be concatenated at the end.
        print('\nApply initial transformation to moving image...')
        sct.run('WarpImageMultiTransform 3 '+file_src_tmp+'.nii '+file_src_reg_tmp+'.nii -R '+file_dest_tmp+'.nii '+fname_init_transfo+' --use-BSpline')
        sct.run('WarpImageMultiTransform 3 '+file_src_seg_tmp+'.nii '+file_src_seg_reg_tmp+'.nii -R '+file_dest_seg_tmp+'.nii '+fname_init_transfo+' --use-BSpline')
        file_src_tmp = file_src_reg_tmp
        file_src_seg_tmp = file_src_seg_reg_tmp

    # Pad the target and source image (because ants doesn't deform the extremities)
    if padding:
        # Pad source image
        print('\nPad source...')
        pad_image(file_src_tmp,file_src_tmp+'_pad.nii',padding)
        file_src_tmp = file_src_tmp+'_pad' # update file name
        # Pad destination image
        print('\nPad destination...')
        pad_image(file_dest_tmp,file_dest_tmp+'_pad.nii',padding)
        file_dest_tmp = file_dest_tmp+'_pad' # update file name
        if use_segmentation:
            # Pad source image
            print('\nPad source segmentation...')
            pad_image(file_src_seg_tmp,file_src_seg_tmp+'_pad.nii',padding)
            file_src_seg_tmp = file_src_seg_tmp+'_pad' # update file name
            # Pad destination image
            print('\nPad destination segmentation...')
            pad_image(file_dest_seg_tmp,file_dest_seg_tmp+'_pad.nii',padding)
            file_dest_seg_tmp = file_dest_seg_tmp+'_pad' # update file name


    # don't use spinal cord segmentation
    if use_segmentation == 0:

        # Estimate transformation using ANTS
        print('\nEstimate transformation using ANTS (might take a couple of minutes)...')

        cmd = 'antsRegistration \
--dimensionality 3 \
'+use_init_transfo+' \
--transform SyN[0.1,3,0] \
--metric MI['+file_dest_tmp+'.nii,'+file_src_tmp+'.nii,1,32] \
--convergence '+numberIterations+' \
--shrink-factors 2x1 \
--smoothing-sigmas 0x0mm \
--Restrict-Deformation 1x1x0 \
--output [tmp.reg,'+file_src_tmp+'_reg.nii] \
--collapse-output-transforms 1 \
--interpolation BSpline[3] \
--winsorize-image-intensities [0.005,0.995]'

        status, output = sct.run(cmd)
        if verbose:
            print output

    # use spinal cord segmentation
    elif use_segmentation == 1:

        ## if use initial transformation (!! needs to be inserted before the --transform field in antsRegistration)
        #if fname_init_transfo != '':
        #    file_src_reg_tmp = file_src_tmp+'_reg'
        #    file_src_seg_reg_tmp = file_src_seg_tmp+'_reg'
        #    # apply initial transformation to moving image, and then estimate transformation between this output and
        #    # destination image. This approach was chosen instead of inputting the transfo into ANTs, because if the transfo
        #    # does not bring the image to the same space as the destination image, then warping fields cannot be concatenated at the end.
        #    print('\nApply initial transformation to moving image...')
        #    #cmd = 'WarpImageMultiTransform 3 '+file_src_tmp+'.nii '+file_src_reg_tmp+'.nii -R '+file_dest_tmp+'.nii '+fname_init_transfo+' --use-BSpline'
        #    sct.run('WarpImageMultiTransform 3 '+file_src_tmp+'.nii '+file_src_reg_tmp+'.nii -R '+file_dest_tmp+'.nii '+fname_init_transfo)
        #    # smooth image
        #    sct.run('c3d tmp.src_pad_reg.nii -smooth 0.5mm -o tmp.src_pad_reg_smooth.nii')
        #    sct.run('WarpImageMultiTransform 3 '+file_src_seg_tmp+'.nii '+file_src_seg_reg_tmp+'.nii -R '+file_dest_seg_tmp+'.nii '+fname_init_transfo)
        #    file_src_tmp = file_src_reg_tmp
        #    file_src_seg_tmp = file_src_seg_reg_tmp
        #    #cmd = 'WarpImageMultiTransform 3 '+file_src_seg_tmp+' '+file_src_seg_reg_tmp+' -R '+file_dest_seg_tmp+' '+fname_init_transfo
        #    #use_init_transfo = ' --initial-moving-transform '+fname_init_transfo
        #    #output_warping_field = "tmp.regSeg1Warp.nii.gz"

        # Estimate transformation using ANTS
        print('\nStep #1: Estimate transformation using spinal cord segmentations...')

        cmd = 'antsRegistration \
--dimensionality 3 \
--transform SyN[0.5,3,0] \
--metric MI['+file_dest_seg_tmp+'.nii,'+file_src_seg_tmp+'.nii,1,32] \
--convergence '+numberIterations+' \
--shrink-factors 4x1 \
--smoothing-sigmas 1x1mm \
--Restrict-Deformation 1x1x0 \
--output [tmp.regSeg,tmp.regSeg.nii]'

#'+use_init_transfo+' \

        #if fname_init_transfo != '':
        #    cmd = cmd+' --initial-moving-transform '+fname_init_transfo
        #    output_warping_field = "tmp.regSeg1Warp.nii.gz"
              
        status, output = sct.run(cmd)
        if verbose:
            print output

        print('\nStep #2: Improve local deformation using images (start from previous transformation)...')

        cmd = 'antsRegistration \
--dimensionality 3 \
--initial-moving-transform '+output_warping_field+' \
--transform SyN[0.1,1,0] \
--metric MI['+file_dest_tmp+'.nii,'+file_src_tmp+'.nii,1,32] \
--convergence 20 \
--shrink-factors 1 \
--smoothing-sigmas 0mm \
--Restrict-Deformation 1x1x0 \
--output [tmp.reg,'+file_src_tmp+'_reg.nii] \
--collapse-output-transforms 1 \
--interpolation BSpline[3]'

        #if fname_init_transfo != '':
        #    cmd = cmd+' --initial-moving-transform '+fname_init_transfo
        
        status, output = sct.run(cmd)
        if verbose:
            print output

    # update file name
    file_src_tmp = file_src_tmp+'_reg'
    file_warp_final = 'tmp.reg0Warp.nii.gz'

    # Concatenate transformations if user had initial transfo
    if fname_init_transfo != '':
        cmd = 'ComposeMultiTransform 3 tmp.reg0WarpConcat.nii.gz -R tmp.dest.nii '+file_warp_final+' '+fname_init_transfo
        print('>> ' + cmd)
        commands.getstatusoutput(cmd)  # here cannot use sct.run() because of wrong output status in ComposeMultiTransform
        file_warp_final = 'tmp.reg0WarpConcat.nii.gz'

    # Apply warping field to src data
    print('\nApply warping field to source data...')
    cmd = 'WarpImageMultiTransform 3 tmp.src.nii tmp.src_reg.nii -R tmp.dest.nii '+file_warp_final+' --use-BSpline'
    status, output = sct.run(cmd)

    ## Remove padding
    #if padding:
    #    print('\nRemove padding...')
    #    remove_padding(fname_dest,file_src_tmp,file_src_tmp+'_nopad.nii')
    #    file_src_tmp = file_src_tmp+'_nopad' # update file name


    # Generate output files
    print('\nGenerate output files...')
#    if fname_init_transfo == '':
    fname_output = sct.generate_output_file('tmp.src_reg.nii', path_out, file_out, ext_out)
    sct.generate_output_file(file_warp_final, path_out, 'warp_src2dest', '.nii.gz')

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm tmp.*')

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

    # to view results
    print '\nTo view results, type:'
    print 'fslview '+fname_dest+' '+fname_output+' &\n'



# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  This program co-registers two spinal cord volumes. The deformation is non-rigid and is constrained in the Z\n' \
        '  direction (i.e., axial plane). Hence, this function assumes that orientation of the DEST image is axial.\n' \
        '  If you need to register two volumes with large deformations and/or different contrasts, it is recommended to\n' \
        '  input spinal cord segmentations (binary mask) in order to achieve maximum robustness. To do so, you can use\n' \
        '  sct_segmentation_propagation.\n' \
        '  The program outputs a warping field that can be used to register other images to the destination image.\n' \
        '  To apply the warping field to another image, type this:\n' \
        '    WarpImageMultiTransform 3 another_image another_image_reg -R dest_image warp_src2dest.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <source> -d <dest>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <source>                  source image\n' \
        '  -d <dest>                    destination image\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -s <source_seg>              spinal cord segmentation for source image (mandatory if -t is used)\n' \
        '  -t <dest_seg>                spinal cord segmentation for destination image (mandatory if -s is used)\n' \
        '  -q <init_transfo>            transformation file (ITK-based) to apply to source image before registration. Default=none''\n' \
        '  -o <output>                  name of output file. Default=source_reg\n' \
        '  -n <N1xN2>                   number of iterations for first and second stage. Default='+param.numberIterations+'\n' \
        '  -p <padding>                 size of padding at top and bottom, to enable deformation at volume edge. Default='+str(param.padding)+'\n' \
        '  -r <0,1>                     remove temporary files. Default='+str(param.remove_temp_files)+'\n' \
        '  -v <0,1>                     verbose. Default='+str(param.verbose)+'\n'


    # exit program
    sys.exit(2)



# pad an image
# ==========================================================================================
def pad_image(fname_in,file_out,padding):
    cmd = 'c3d '+fname_in+' -pad 0x0x'+str(padding)+'vox 0x0x'+str(padding)+'vox 0 -o '+file_out
    print(">> "+cmd)
    os.system(cmd)
    return



# remove padding
# ==========================================================================================
def remove_padding(file_ref,file_in,file_out):
    # remove padding by reslicing padded data into unpadded space
    cmd = 'c3d '+file_ref+' '+file_in+' -reslice-identity -o '+file_out
    print(">> "+cmd)    
    os.system(cmd)
    return



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
    #    cmd = 'c3d -mcs tmp.regWarp.nii -oo tmp.regWarp_x.nii tmp.regWarp_y.nii tmp.regWarp_z.nii'
    #    print(">> "+cmd)
    #    os.system(cmd)
    #    cmd = 'fslmerge -t '+path_out+'warp_comp.nii tmp.regWarp_x.nii tmp.regWarp_y.nii tmp.regWarp_z.nii'
    #    print(">> "+cmd)
    #    os.system(cmd)
    #===========
