#!/usr/bin/env python
#########################################################################################
#
# Detect vertebral levels from centerline.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Eugenie Ullmann, Karun Raju, Tanguy Duval, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# check if needed Python libraries are already installed or not

# from msct_base_classes import BaseScript
import sys
from os import chdir
from time import strftime

import numpy as np

from sct_utils import extract_fname, printv, run, generate_output_file, slash_at_the_end
from msct_parser import Parser
from msct_image import Image


class Param:
    def __init__(self):
        self.verbose = '1'
        self.remove_tmp_files = '1'


# class Script(BaseScript):
#     def __init__(self):
#         super(Script, self).__init__()
#
#     @staticmethod


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)
    parser.usage.set_description('''This program automatically detect the spinal cord in a MR image and output a centerline of the spinal cord.''')
    parser.add_option(name="-i",
                      type_value="file",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-seg",
                      type_value="file",
                      description="Segmentation or centerline of the spinal cord.",
                      mandatory=True,
                      example="t2_seg.nii.gz")
    parser.add_option(name="-t",
                      type_value="multiple_choice",
                      description="Image contrast: t2: cord dark / CSF bright ; t1: cord bright / CSF dark",
                      mandatory=True,
                      example=["t1", "t2"])
    # parser.add_option(name="-seg",
    #                   type_value="file",
    #                   description="input image.",
    #                   mandatory=True,
    #                   example="segmentation.nii.gz")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files.",
                      mandatory=False,
                      default_value=param.remove_tmp_files,
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=param.verbose,
                      example=['0', '1', '2'])
    parser.add_option(name="-h",
                      type_value=None,
                      description="display this help",
                      mandatory=False)
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments["-i"]
    fname_seg = arguments['-seg']
    contrast = arguments['-t']
    if '-o' in arguments:
        fname_out = arguments["-o"]
    else:
        fname_out = ''
    param.verbose = int(arguments['-v'])
    param.remove_tmp_files = int(arguments['-r'])

    # create temporary folder
    printv('\nCreate temporary folder...', param.verbose)
    path_tmp = slash_at_the_end('tmp.'+strftime("%y%m%d%H%M%S"), 1)
    run('mkdir '+path_tmp, param.verbose)

    # Copying input data to tmp folder
    printv('\nCopying input data to tmp folder...', param.verbose)
    run('sct_convert -i '+fname_in+' -o '+path_tmp+'data.nii')
    run('sct_convert -i '+fname_seg+' -o '+path_tmp+'segmentation.nii.gz')

    # Go go temp folder
    chdir(path_tmp)

    # Straighten spinal cord
    printv('\nStraighten spinal cord...', param.verbose)
    run('sct_straighten_spinalcord -i data.nii -c segmentation.nii.gz')

    # Apply straightening to segmentation
    # N.B. Output is RPI
    printv('\nApply straightening to segmentation...', param.verbose)
    run('sct_apply_transfo -i segmentation.nii.gz -d data_straight.nii -w warp_curve2straight.nii.gz -o segmentation_straight.nii.gz -x linear')
    # Threshold segmentation to 0.5
    run('sct_maths -i segmentation_straight.nii.gz -thr 0.5 -o segmentation_straight.nii.gz')

    init_disk = [144, 5]
    # detect vertebral levels on straight spinal cord
    vertebral_detection('data_straight.nii', 'segmentation_straight.nii.gz', contrast, init_disk)

    # un-straighten spinal cord
    printv('\nUn-straighten labeling...', param.verbose)
    run('sct_apply_transfo -i segmentation_straight_labeled.nii.gz -d segmentation.nii.gz -w warp_straight2curve.nii.gz -o segmentation_labeled.nii.gz -x nn')

    # clean labeled segmentation
    # TODO: (i) find missing voxels wrt. original segmentation, and attribute value closest to neighboring label and (ii) remove additional voxels.

    # Build fname_out
    if fname_out == '':
        path_seg, file_seg, ext_seg = extract_fname(fname_seg)
        fname_out = path_seg+file_seg+'_labeled'+ext_seg

    # come back to parent folder
    chdir('..')

    # Generate output files
    printv('\nGenerate output files...', param.verbose)
    generate_output_file(path_tmp+'segmentation_labeled.nii.gz', fname_out)

    # Remove temporary files
    if param.remove_tmp_files == 1:
        printv('\nRemove temporary files...', param.verbose)
        run('rm -rf '+path_tmp)

    # to view results
    printv('\nDone! To view results, type:', param.verbose)
    printv('fslview '+fname_in+' '+fname_out+' &\n', param.verbose, 'info')



# Detect vertebral levels
# ==========================================================================================
def vertebral_detection(fname, fname_seg, contrast, init_disk):

    shift_AP = 15  # shift the centerline towards the spine (in mm).
    size_AP = 5  # mean around the centerline in the anterior-posterior direction in mm
    size_RL = 7  # mean around the centerline in the right-left direction in mm
    verbose = param.verbose

    if verbose == 2:
        import matplotlib.pyplot as plt
        plt.ion()  # enables interactive mode

    # open anatomical volume
    img = Image(fname)
    # orient to RPI
    # img.change_orientation()
    # get dimension
    nx, ny, nz, nt, px, py, pz, pt = img.dim

    # matshow(img.data[:, :, 100]), show()

    #==================================================
    # Compute intensity profile across vertebrae
    #==================================================

    shift_AP = shift_AP * py
    size_AP = size_AP * py
    size_RL = size_RL * px

    # define z: vector of indices along spine
    z = range(nz)

    # define xc and yc (centered in the field of view)
    xc = round(nx/2)  # direction RL
    yc = round(ny/2)  # direction AP
    I = np.zeros((nz, 1))
    for iz in range(nz):
        vox_in_spine = np.mgrid[xc-size_RL:xc+size_RL+1, yc+shift_AP-size_AP:yc+shift_AP+size_AP+1]
        # average intensity within box in the spine (shifted from spinal cord)
        I[iz] = np.mean(img.data[vox_in_spine[0, :, :].ravel().astype(int),
                                 vox_in_spine[1, :, :].ravel().astype(int),
                                 iz])

    # display intensity along spine
    if verbose == 2:
        plt.figure()
        plt.plot(I)
        plt.title('Averaged intensity within spine. x=0: most caudal.')
        plt.draw()

    # find local extrema
    from scipy.signal import argrelextrema
    peaks = argrelextrema(I, np.greater, order=10)[0]
    nb_peaks = len(peaks)

    if verbose == 2:
        plt.figure()
        plt.plot(I)
        plt.plot(peaks, I[peaks], 'ro')
        plt.draw()

    # LABEL PEAKS
    # build labeled peak vector (inverted order because vertebral level decreases when z increases)
    labeled_peaks = np.array(range(nb_peaks+1, 1, -1)).astype(int)
    # find peak index closest to user input
    peak_ind_closest = np.argmin(abs(peaks-init_disk[0]))
    # build vector of peak labels
    # labeled_peaks = np.array(range(nb_peaks))
    # add the difference between "peak_ind_closest" and the init_disk value
    labeled_peaks = labeled_peaks - peak_ind_closest + init_disk[1]

    # REMOVE WRONG LABELS (ASSUMING NO PEAK IS VISIBLE ABOVE C2/C3 DISK)
    ind_true_labels = np.where(labeled_peaks>1)[0]
    peaks = peaks[ind_true_labels]
    labeled_peaks = labeled_peaks[ind_true_labels]

    # ADD C1 label (ASSUMING DISTANCE FROM THE ADULT TEMPLATE)
    distance_c1_c2 = 20.8300/pz  # in mm
    # check if C2 disk is there
    if np.min(labeled_peaks) == 2:
        printv('\nC2 disk is present. Adding C1 labeling based on template.')
        peaks = np.append(peaks, (np.max(peaks) + distance_c1_c2).astype(int))
        labeled_peaks = np.append(labeled_peaks, 1)

    # LABEL SEGMENTATION
    # open segmentation
    seg = Image(fname_seg)
    for iz in range(nz):
        # get value of the disk above iz
        ind_above_iz = np.nonzero((peaks-iz).clip(0))[0]
        if not ind_above_iz.size:
            # if ind_above_iz is empty, attribute value 0
            # vertebral_level = np.min(labeled_peaks)
            vertebral_level = 0
        else:
            # ind_disk_above = np.where(peaks-iz > 0)[0][0]
            ind_disk_above = min(ind_above_iz)
            # assign vertebral level (remove one because iz is BELOW the disk)
            vertebral_level = labeled_peaks[ind_disk_above] + 1
            # print vertebral_level
        # get voxels in mask
        ind_nonzero = np.nonzero(seg.data[:, :, iz])
        seg.data[ind_nonzero[0], ind_nonzero[1], iz] = vertebral_level

    # WRITE LABELED SEGMENTATION
    seg.file_name += '_labeled'
    seg.save()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # # initialize parameters
    param = Param()
    # call main function
    main()
