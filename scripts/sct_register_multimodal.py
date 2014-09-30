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

# TODO: testing script for all cases
# TODO: try to combine seg and image based for 2nd stage
# TODO: output name file for warp using "src" and "dest" file name, i.e. warp_filesrc2filedest.nii.gz
# TODO: set gradient-step-length in mm instead of vox size.

# Note for the developer: DO NOT use --collapse-output-transforms 1, otherise inverse warping field is not output


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug               = 0
        self.remove_temp_files   = 1 # remove temporary files
        self.outSuffix           = "_reg"
        self.padding             = 3 # add 'padding' slices at the top and bottom of the volumes if deformation at the edge is not good. Default=5. Put 0 for no padding.
#        self.convertDeformation  = 0 # Convert deformation field to 4D volume (readable by fslview)
        self.algo = 'SyN'
        self.numberIterations    = "15x3" # number of iterations
        self.numberIterationsStep2 = "10" # number of iterations at step 2
        self.verbose             = 0 # verbose
        self.compute_dest2sr     = 0 # compute dest2src warping field
        self.gradientStep_input = '0.2,0.5'  # gradientStep in SyN transformation. First value is for image-based, second is for segmentation-based (if exist)

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
    fname_init_transfo = ''
    fname_init_transfo_inv = ''
    use_init_transfo = ''
    gradientStep_input = param.gradientStep_input
    compute_dest2src = param.compute_dest2sr
    algo = param.algo
    start_time = time.time()
    print ''

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        #fname_src = path_sct+'/data/template/MNI-Poly-AMU_T2.nii.gz'
        #fname_src = path_sct+'/testing/data/errsm_23/mt/mtc0.nii.gz'
        #fname_dest = path_sct+'/testing/data/errsm_23/mt/mtc1.nii.gz'
        fname_src = '/Users/julien/data/errsm/errsm_30/t2/template2anat.nii.gz'
        fname_dest = '/Users/julien/data/errsm/errsm_30/t1/t1.nii.gz'
        #fname_src_seg = path_sct+'/data/template/MNI-Poly-AMU_cord.nii.gz'
        #fname_dest_seg = path_sct+'/testing/data/errsm_23/mt/segmentation_binary.nii.gz'
        #fname_init_transfo = path_sct+'/testing/data/errsm_23/template/warp_template2anat.nii.gz'
        #fname_init_transfo_inv = path_sct+'/testing/data/errsm_23/template/warp_anat2template.nii.gz'
        numberIterations = '3x0'
        numberIterationsStep2 = "1"
        gradientStep_input = '0.2, 0.5'
        compute_dest2src = 1
        verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hd:g:i:m:n:o:p:q:r:s:t:v:x:y:z:')
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
        elif opt in ('-x'):
            compute_dest2src = int(arg)
        elif opt in ('-y'):
            numberIterationsStep2 = arg
        elif opt in ('-z'):
            fname_init_transfo_inv = arg

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

    # Parse gradient step
    print '\nParse gradient step...'
    gradientStep = []
    gradientStep_input = gradientStep_input.replace(' ', '')  # remove spaces
    gradientStep_input = gradientStep_input.split(",")  # parse with comma
    for i in range(len(gradientStep_input)):
        try:
            float(gradientStep_input[i])
            gradientStep.append(gradientStep_input[i])
        except:
            sct.printv('  WARNING: Not a float. Use default value for gradientStep['+str(i)+']', 1, 'warning')

    # print arguments
    print '\nInput parameters:'
    print '  Source .............. '+fname_src
    print '  Destinationf ........ '+fname_dest
    print '  Segmentation source . '+fname_src_seg
    print '  Segmentation dest ... '+fname_dest_seg
    print '  Init transfo ........ '+fname_init_transfo
    print '  Output name ......... '+fname_output
    print '  Algorithm ........... '+algo
    print '  Iterations at step1 (seg) .... '+str(numberIterations)
    print '  Iterations at step2 (image) .. '+str(numberIterationsStep2)
    print '  Gradient step ....... '+str(gradientStep)
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
    if not fname_init_transfo == '':
        fname_init_transfo = os.path.abspath(fname_init_transfo)  # test if not empty, otherwise it will transform the empty string into a string with path, which is a problem because the emptiness of the string is tested later.
    if not fname_init_transfo_inv == '':
        fname_init_transfo_inv = os.path.abspath(fname_init_transfo_inv)
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
    file_src_tmp = 'src'
    file_dest_tmp = 'dest'
    status, output = sct.run('sct_c3d '+fname_src+' -o '+path_tmp+'/'+file_src_tmp+'.nii')
    status, output = sct.run('sct_c3d '+fname_dest+' -o '+path_tmp+'/'+file_dest_tmp+'.nii')
    if use_segmentation:
        file_src_seg_tmp = 'src_seg'
        file_dest_seg_tmp = 'dest_seg'
        status, output = sct.run('sct_c3d '+fname_src_seg+' -o '+path_tmp+'/'+file_src_seg_tmp+'.nii')
        status, output = sct.run('sct_c3d '+fname_dest_seg+' -o '+path_tmp+'/'+file_dest_seg_tmp+'.nii')

    # go to tmp folder
    os.chdir(path_tmp)

    # Find orientation of source data
    print('\nFind orientation of source data...')
    orientation = sct.get_orientation('src.nii')
    sct.printv('  '+orientation, verbose)

    # Find the dimension corresponding to z
    sct.printv('\nFind the dimension corresponding to the superior-inferior direction...', verbose)
    dimension_si = 0
    if orientation.find('I') != '-1':
        dimension_si = orientation.find('I')
    elif orientation.find('S') != '-1':
        dimension_si = orientation.find('S')
    else:
        sct.printv("ERROR: Cannot find proper dimension. Exit program.", 1, 'error')
        sys.exit(2)
    sct.printv('  '+str(dimension_si), verbose)

    # Adjust ANTs variable so that the deformation is restricted in the slice plane
    restrict_deformation = '1x1x1'
    if dimension_si == 0:
        restrict_deformation = '0x1x1'
    elif dimension_si == 1:
        restrict_deformation = '1x0x1'
    elif dimension_si == 2:
        restrict_deformation = '1x1x0'
    else:
        sct.printv("ERROR: Cannot adjust variable: restrict_deformation. Exit program.", 1, 'error')
        sys.exit(2)

    # if use initial transformation (!! needs to be inserted before the --transform field in sct_antsRegistration)
    if fname_init_transfo != '':
        file_src_reg_tmp = file_src_tmp+'_reg'
        # apply initial transformation to moving image, and then estimate transformation between this output and
        # destination image. This approach was chosen instead of inputting the transfo into ANTs, because if the transfo
        # does not bring the image to the same space as the destination image, then warping fields cannot be concatenated at the end.
        print('\nApply initial transformation to moving image...')
        sct.run('sct_WarpImageMultiTransform 3 '+file_src_tmp+'.nii '+file_src_reg_tmp+'.nii -R '+file_dest_tmp+'.nii '+fname_init_transfo+' --use-BSpline')
        file_src_tmp = file_src_reg_tmp
        if use_segmentation:
            file_src_seg_reg_tmp = file_src_seg_tmp+'_reg'
            sct.run('sct_WarpImageMultiTransform 3 '+file_src_seg_tmp+'.nii '+file_src_seg_reg_tmp+'.nii -R '+file_dest_seg_tmp+'.nii '+fname_init_transfo+' --use-BSpline')
            file_src_seg_tmp = file_src_seg_reg_tmp

    # Pad the target and source image (because ants doesn't deform the extremities)
    if padding:
        # Pad source image
        print('\nPad source...')
        pad_image(file_src_tmp, file_src_tmp+'_pad.nii', padding)
        file_src_tmp += '_pad'  # update file name
        # Pad destination image
        print('\nPad destination...')
        pad_image(file_dest_tmp, file_dest_tmp+'_pad.nii', padding)
        file_dest_tmp += '_pad'  # update file name
        if use_segmentation:
            # Pad source image
            print('\nPad source segmentation...')
            file_src_seg_tmp = file_src_seg_tmp
            pad_image(file_src_seg_tmp, file_src_seg_tmp+'_pad.nii', padding)
            file_src_seg_tmp += '_pad'  # update file name
            # Pad destination image
            print('\nPad destination segmentation...')
            pad_image(file_dest_seg_tmp, file_dest_seg_tmp+'_pad.nii', padding)
            file_dest_seg_tmp += '_pad'  # update file name

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

        # Estimate transformation using ANTS
        print('\nStep #1: Estimate transformation using spinal cord segmentations...')

        cmd = 'sct_antsRegistration \
--dimensionality 3 \
--transform '+algo+'['+str(gradientStep[1])+',3,0] \
--metric MI['+file_dest_seg_tmp+'.nii,'+file_src_seg_tmp+'.nii,1,32] \
--convergence '+numberIterations+' \
--shrink-factors 4x1 \
--smoothing-sigmas 1x1mm \
--Restrict-Deformation '+restrict_deformation+' \
--output [regSeg,regSeg.nii]'

        status, output = sct.run(cmd)
        if verbose:
            print output

        print('\nStep #2: Improve local deformation using images (start from previous transformation)...')

        cmd = 'sct_antsRegistration \
--dimensionality 3 \
--initial-moving-transform regSeg0Warp.nii.gz \
--transform '+algo+'['+str(gradientStep[0])+',3,0] \
--metric MI['+file_dest_tmp+'.nii,'+file_src_tmp+'.nii,1,32] \
--convergence '+numberIterationsStep2+' \
--shrink-factors 1 \
--smoothing-sigmas 0mm \
--Restrict-Deformation '+restrict_deformation+' \
--output [reg,'+file_src_tmp+'_reg.nii] \
--collapse-output-transforms 0 \
--interpolation BSpline[3]'

        status, output = sct.run(cmd)
        if verbose:
            print output

    # Concatenate transformations
    print('\nConcatenate transformations...')

    # if user has initial transfo:
    if fname_init_transfo != '':
        if use_segmentation == 0:
            # src --> dest
            cmd1 = 'sct_ComposeMultiTransform 3 warp_src2dest.nii.gz -R dest.nii reg0Warp.nii.gz '+fname_init_transfo
            # dest --> src
            if compute_dest2src:
                cmd2 = 'sct_ComposeMultiTransform 3 warp_dest2src.nii.gz -R src.nii '+fname_init_transfo_inv+' reg0InverseWarp.nii.gz'

        elif use_segmentation == 1:
            # src --> dest
            cmd1 = 'sct_ComposeMultiTransform 3 warp_src2dest.nii.gz -R dest.nii reg1Warp.nii.gz regSeg0Warp.nii.gz '+fname_init_transfo
            # dest --> src
            if compute_dest2src:
                cmd2 = 'sct_ComposeMultiTransform 3 warp_dest2src.nii.gz -R src.nii '+fname_init_transfo_inv+' regSeg0InverseWarp.nii.gz reg1InverseWarp.nii.gz'

    # if user does not have initial transfo:
    else:
        if use_segmentation == 0:
            # src --> dest
            cmd1 = 'sct_ComposeMultiTransform 3 warp_src2dest.nii.gz -R dest.nii reg0Warp.nii.gz'
            # dest --> src
            if compute_dest2src:
                cmd2 = 'sct_ComposeMultiTransform 3 warp_dest2src.nii.gz -R src.nii reg0InverseWarp.nii.gz'

        elif use_segmentation == 1:
            # src --> dest
            cmd1 = 'sct_ComposeMultiTransform 3 warp_src2dest.nii.gz -R dest.nii reg1Warp.nii.gz regSeg0Warp.nii.gz'
            # dest --> src
            if compute_dest2src:
                cmd2 = 'sct_ComposeMultiTransform 3 warp_dest2src.nii.gz -R src.nii regSeg0InverseWarp.nii.gz reg1InverseWarp.nii.gz'

    print('>> ' + cmd1)
    commands.getstatusoutput(cmd1)  # here cannot use sct.run() because of wrong output status in sct_ComposeMultiTransform
    if compute_dest2src:
        print('>> ' + cmd2)
        commands.getstatusoutput(cmd2)  # here cannot use sct.run() because of wrong output status in sct_ComposeMultiTransform

    # Apply warping field to src data
    print('\nApply transfo source --> dest...')
    status, output = sct.run('sct_WarpImageMultiTransform 3 src.nii src_reg.nii -R dest.nii warp_src2dest.nii.gz --use-BSpline')
    if compute_dest2src:
        print('\nApply transfo dest --> source...')
        status, output = sct.run('sct_WarpImageMultiTransform 3 dest.nii dest_reg.nii -R src.nii warp_dest2src.nii.gz --use-BSpline')

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    print('\nGenerate output files...')
    fname_src2dest = sct.generate_output_file(path_tmp+'/src_reg.nii', path_out+file_out+ext_out)
    sct.generate_output_file(path_tmp+'/warp_src2dest.nii.gz', path_out+'warp_src2dest'+'.nii.gz')
    if compute_dest2src:
        fname_dest2src = sct.generate_output_file(path_tmp+'/dest_reg.nii', path_out+file_dest+'_reg'+ext_dest)
        sct.generate_output_file(path_tmp+'/warp_dest2src.nii.gz', path_out+'warp_dest2src.nii.gz')

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
    if compute_dest2src:
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
  image. To apply the warping field to another image, type this:
    sct_WarpImageMultiTransform 3 another_image another_image_reg -R dest_image warp_src2dest.

USAGE
  """+os.path.basename(__file__)+""" -i <source> -d <dest>

MANDATORY ARGUMENTS
  -i <source>                  source image
  -d <dest>                    destination image

OPTIONAL ARGUMENTS
  -s <source_seg>              segmentation for source image (mandatory if -t is used)
  -t <dest_seg>                segmentation for destination image (mandatory if -s is used)
  -q <init_transfo>            transformation file (ITK-based) to apply to source image before
                               registration. Default=none
  -x {0,1}                     compute inverse transformation (dest --> source)
  -z <init_transfo_inv>        inverse transformation file to obtain warp_dest2src deformation field
                               N.B. Only use this flag with -q and -x
  -o <output>                  name of output file. Default=source_reg
  -n <N1xN2>                   number of iterations for first and second stage. Default="""+param.numberIterations+"""
  -y <N>                       number of iterations at step 2 (if using segmentation). Set 0 to register based on segmentation only. Default="""+param.numberIterationsStep2+"""
  -g <val1,val2>               gradientStep for SyN transformation. The larger the more deformation.
                               If you use a segmentation, you can specify gradientStep for each
                               step as follow: val1,val2 (val1: image, val2: seg).
                               Default="""+param.gradientStep_input+"""
  -p <padding>                 size of padding (top & bottom), to enable deformation at edges.
                               Default="""+str(param.padding)+"""
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
def pad_image(fname_in,file_out,padding):
    cmd = 'sct_c3d '+fname_in+' -pad 0x0x'+str(padding)+'vox 0x0x'+str(padding)+'vox 0 -o '+file_out
    print(">> "+cmd)
    os.system(cmd)
    return



# remove padding
# ==========================================================================================
def remove_padding(file_ref,file_in,file_out):
    # remove padding by reslicing padded data into unpadded space
    cmd = 'sct_c3d '+file_ref+' '+file_in+' -reslice-identity -o '+file_out
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
    #    cmd = 'sct_c3d -mcs tmp.regWarp.nii -oo tmp.regWarp_x.nii tmp.regWarp_y.nii tmp.regWarp_z.nii'
    #    print(">> "+cmd)
    #    os.system(cmd)
    #    cmd = 'fslmerge -t '+path_out+'warp_comp.nii tmp.regWarp_x.nii tmp.regWarp_y.nii tmp.regWarp_z.nii'
    #    print(">> "+cmd)
    #    os.system(cmd)
    #===========
