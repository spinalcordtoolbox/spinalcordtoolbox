#!/usr/bin/env python
#########################################################################################
#
# Validation script for SCAD (Spinal Cord Automatic Detection)
#
# Brainhack MTL 2015: Algorithms for automatic spinal cord detection on MR images
#
# This repository is intented to develop and test new algorithms for automatically detect the spinal cord on various
# contrasts of MR volumes.
# The developed algorithms must satisfy the following criteria:
# - they can be coded in Python or C++
# - they must read a nifti image as input image (.nii or .nii.gz): "-i" (input file name) option
# - they have to output a binary image with the same format and orientation as input image, containing the location
#   or the centerline of the spinal cord: "-o" (output file name) option
# - they have to be **fast**
#
# To validate a new algorithm, it must go through the validation pipeline using the following command:
#
# scad_validation.py "algo_name"
#
# The validation pipeline tests your algorithm throughout a testing dataset, containing many images of the spinal cord
# with various contrasts and fields of view, along with their manual segmentation.
# It tests several criteria:
# 1. if your detection is inside the spinal cord
# 2. if your detection is near the spinal cord centerline (at least near the manual centerline)
# 3. if the length of the centerline your algorithm extracted correspond with the manual centerline
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2015-07-22
#
# About the license: see the file LICENSE
#########################################################################################

import sys
import os
import nibabel as nib
from msct_image import Image
from dev.sct_scad import SCAD
import numpy as np
import sct_utils as sct
from sct_utils import printv
import matplotlib.pyplot as plt


def scadMRValidation(algorithm, isPython=False, verbose=True):
    if not isinstance(algorithm, str) or not algorithm:
        print 'ERROR: You must provide the name of your algorithm as a string.'
        usage()

    import time
    import sct_utils as sct

    # creating a new folder with the experiment
    path_experiment = 'scad-experiment.'+algorithm+'.'+time.strftime("%y%m%d%H%M%S")
    #status, output = sct.run('mkdir '+path_experiment, verbose)

    # copying images from "data" folder into experiment folder
    sct.copyDirectory('data', path_experiment)

    # Starting validation
    os.chdir(path_experiment)
    # t1
    os.chdir('t1/')
    for subject_dir in os.listdir('./'):
        if os.path.isdir(subject_dir):
            os.chdir(subject_dir)

            # creating list of images and corresponding manual segmentation
            list_images = dict()
            for file_name in os.listdir('./'):
                if not 'manual_segmentation' in file_name:
                    for file_name_corr in os.listdir('./'):
                        if 'manual_segmentation' in file_name_corr and sct.extract_fname(file_name)[1] in file_name_corr:
                            list_images[file_name] = file_name_corr

            # running the proposed algorithm on images in the folder and analyzing the results
            for image, image_manual_seg in list_images.items():
                print image
                path_in, file_in, ext_in = sct.extract_fname(image)
                image_output = file_in+'_centerline'+ext_in
                if ispython:
                    try:
                        eval(algorithm+'('+image+', t1, verbose='+str(verbose)+')')
                    except Exception as e:
                        print 'Error during spinal cord detection on line {}:'.format(sys.exc_info()[-1].tb_lineno)
                        print 'Subject: t1/'+subject_dir+'/'+image
                        print e
                        sys.exit(2)
                else:
                    cmd = algorithm+' -i '+image+' -t t1'
                    if verbose:
                        cmd += ' -v'
                    status, output = sct.run(cmd, verbose=verbose)
                    if status != 0:
                        print 'Error during spinal cord detection on Subject: t1/'+subject_dir+'/'+image
                        print output
                        sys.exit(2)

                # analyzing the resulting centerline
                from msct_image import Image
                manual_segmentation_image = Image(image_manual_seg)
                manual_segmentation_image.change_orientation()
                centerline_image = Image(image_output)
                centerline_image.change_orientation()

                from msct_types import Coordinate
                # coord_manseg = manual_segmentation_image.getNonZeroCoordinates()
                coord_centerline = centerline_image.getNonZeroCoordinates()

                # check if centerline is in manual segmentation
                result_centerline_in = True
                for coord in coord_centerline:
                    if manual_segmentation_image.data[coord.x, coord.y, coord.z] == 0:
                        result_centerline_in = False
                        print 'failed on slice #' + str(coord.z)
                        break
                if result_centerline_in:
                    print 'OK: Centerline is inside manual segmentation.'
                else:
                    print 'FAIL: Centerline is outside manual segmentation.'


                # check the length of centerline compared to manual segmentation
                # import sct_process_segmentation as sct_seg
                # length_manseg = sct_seg.compute_length(image_manual_seg)
                # length_centerline = sct_seg.compute_length(image_output)
                # if length_manseg*0.9 <= length_centerline <= length_manseg*1.1:
                #     print 'OK: Length of centerline correspond to length of manual segmentation.'
                # else:
                #     print 'FAIL: Length of centerline does not correspond to length of manual segmentation.'
            os.chdir('..')

    # t2

    # t2*

    # dmri

    # gre

def validate_scad(folder_input, contrast):
    """
    Expecting folder to have the following structure :
    errsm_01:
    - t2
    -- errsm_01.nii.gz or t2.nii.gz
    --
    :param folder_input:
    :return:
    """
    from sct_get_centerline import ind2sub
    import time
    import math
    import numpy

    t0 = time.time()

    current_folder = os.getcwd()
    os.chdir(folder_input)

    try:
        patients = next(os.walk('.'))[1]
        overall_distance = {}
        max_distance = {}
        standard_deviation = 0
        overall_std = {}
        rmse = {}
        for i in patients:
            directory = i + "/" + str(contrast)
            try:
                os.chdir(directory)
            except Exception, e:
                print str(i)+" : "+contrast+" directory not found"

            try:
                if os.path.isfile(i+"_"+contrast+".nii.gz"):
                    raw_image = Image(i+"_"+contrast+".nii.gz")
                elif os.path.isfile(contrast+".nii.gz"):
                    raw_image = Image(contrast+".nii.gz")
                else:
                    raise Exception("Patient scan not found")

                if os.path.isfile(i+"_"+contrast+"_manual_segmentation.nii.gz"):
                    raw_orientation = raw_image.change_orientation()
                    scad = SCAD(raw_image, contrast=contrast, rm_tmp_file=1, verbose=1)
                    scad.execute()

                    manual_seg = Image(i+"_"+contrast+"_manual_segmentation.nii.gz")
                    manual_orientation = manual_seg.change_orientation()

                    from scipy.ndimage.measurements import center_of_mass
                    # find COM
                    iterator = range(manual_seg.data.shape[2])
                    com_x = [0 for ix in iterator]
                    com_y = [0 for iy in iterator]

                    for iz in iterator:
                        com_x[iz], com_y[iz] = center_of_mass(manual_seg.data[:, :, iz])

                    centerline_scad = Image(i+"_"+contrast+"_centerline.nii.gz")
                    # os.remove(i+"_"+contrast+"_centerline.nii.gz")

                    centerline_scad.change_orientation()
                    distance = {}
                    for iz in range(1, centerline_scad.data.shape[2]-1):
                        ind1 = np.argmax(centerline_scad.data[:, :, iz])
                        X,Y = ind2sub(centerline_scad.data[:, :, iz].shape,ind1)
                        com_phys = np.array(manual_seg.transfo_pix2phys([[com_x[iz], com_y[iz], iz]]))
                        scad_phys = np.array(centerline_scad.transfo_pix2phys([[X, Y, iz]]))
                        distance_magnitude = np.linalg.norm([com_phys[0][0]-scad_phys[0][0], com_phys[0][1]-scad_phys[0][1], 0])
                        if math.isnan(distance_magnitude):
                            print "Value is nan"
                        else:
                            distance[iz] = distance_magnitude

                    f = open(i+"_"+contrast+"_results.txt", 'w+')
                    f.write("Patient,Slice,Distance")
                    for key, value in distance.items():
                        f.write(i+","+str(key)+","+str(value))

                    standard_deviation = np.std(np.array(distance.values()))
                    average = sum(distance.values())/len(distance)
                    root_mean_square = np.sqrt(np.mean(np.square(distance.values())))

                    f.write("\nAverage : "+str(average))
                    f.write("\nStandard Deviation : "+str(standard_deviation))

                    f.close()

                    overall_distance[i] = average
                    max_distance[i] = max(distance.values())
                    overall_std[i] = standard_deviation
                    rmse[i] = root_mean_square

                else:
                    printv("Cannot find the manual segmentation", type="warning")

            except Exception, e:
                print e.message

            os.chdir(folder_input)

        print time.time() - t0
        # Get overall distance
        f = open("scad_validation_results_"+contrast+"_"+time.strftime("%y%m%d%H%M%S")+".txt", "w+")
        f.write("Patient,Average,MRSE,Standard Deviation,Max\n")
        for key,value in overall_distance.items():
            for subject, max_value in max_distance.items():
                for std_id, std_val in overall_std.items():
                    for rms_i, rms_val in rmse.items():
                        if key == subject and subject == std_id and std_id == rms_i:
                            f.write(key+","+str(value)+","+str(rms_val)+","+str(std_val)+","+str(max_value)+"\n")
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist()


        # average_total = sum(overall_distance.values())/len(overall_distance)
        # average_max = sum(max_distance.values())/len(max_distance)
        #
        # f.write("\n\nTotal average,Average of maximums\n")
        # f.write(str(average_total)+","+str(average_max))
        f.close()

    except Exception, e:
        print e.message


def scad_propseg_validation(folder_input, contrast):
    from sct_get_centerline import ind2sub
    import time
    import math
    import numpy
    import sct_convert as cnv

    t0 = time.time()

    current_folder = os.getcwd()
    os.chdir(folder_input)

    try:
        patients = next(os.walk('.'))[1]
        for i in patients:
            directory = i + "/" + str(contrast)
            try:
                os.chdir(directory)
            except Exception, e:
                print str(i)+" : "+contrast+" directory not found"
            try:
                if os.path.isfile(i+"_"+contrast+".nii.gz"):
                    raw_image = Image(i+"_"+contrast+".nii.gz")
                elif os.path.isfile(contrast+".nii.gz"):
                    raw_image = Image(contrast+".nii.gz")
                else:
                    raise Exception("Patient scan not found")

                if os.path.isfile(i+"_"+contrast+"_manual_segmentation.nii.gz"):
                    manual_segmentation  = i+"_"+contrast+"_manual_segmentation.nii.gz"
                    # Using propseg default
                    sct.run("sct_propseg -i "+raw_image.absolutepath+" -t "+contrast)
                    cnv.convert(raw_image.file_name+"_seg.nii.gz", "propseg_default.nii.gz")
                    # Using scad
                    scad = SCAD(raw_image, contrast=contrast, rm_tmp_file=1, verbose=1)
                    scad.execute()
                    # Using propseg with scad
                    sct.run("sct_propseg -i "+raw_image.absolutepath+" -t "+contrast+" -init-centerline "+scad.output_filename)
                    cnv.convert(raw_image.file_name+"_seg.nii.gz", "propseg_scad.nii.gz")
                    # Calculate dice of propseg_default
                    sct.run("sct_dice_coefficient propseg_default.nii.gz "+manual_segmentation+" -o propseg_default_result.txt")
                    # Calculate dice of propseg_scad
                    sct.run("sct_dice_coefficient propseg_scad.nii.gz "+manual_segmentation+" -o propseg_scad_result.txt")
                else:
                    printv("Cannot find the manual segmentation", type="warning")

            except Exception, e:
                print e.message

            os.chdir(folder_input)
    except Exception, e:
        print e


def check_dices(folder_input, contrast):
    from sct_scad import ind2sub
    import time
    import math
    import numpy
    import sct_convert as cnv
    import matplotlib.pyplot as plt

    t0 = time.time()

    current_folder = os.getcwd()
    os.chdir(folder_input)

    diff = {}

    try:
        patients = next(os.walk('.'))[1]
        for i in patients:
            directory = i + "/" + str(contrast)
            try:
                os.chdir(directory)
            except Exception, e:
                print str(i)+" : "+contrast+" directory not found"
            try:
                if os.path.isfile("propseg_default_result.txt") and os.path.isfile("propseg_scad_result.txt"):
                    propseg_dice = ""
                    scad_dice = ""
                    with open("propseg_default_result.txt", "r+") as f:
                        propseg_dice = f.readlines()
                        f.close()
                    with open("propseg_scad_result.txt", "r+") as f:
                        scad_dice = f.readlines()
                        f.close()

                    diff[i] = float(propseg_dice[0].replace("3D Dice coefficient = ", "")) - float(scad_dice[0].replace("3D Dice coefficient = ", ""))

                else:
                    printv("Cannot find the dice results for patient", type="warning")

            except Exception, e:
                print e.message

            os.chdir(folder_input)

        f = open("dice_results_"+contrast+".txt", "w+")
        f.write("Patient,Dice Difference(default-scad)\n")
        for key, value in diff.items():
            f.write(key+","+str(value)+"\n")

        f.close()

        fig = plt.figure()
        plt.hist(diff.values())
        plt.savefig("Histogram_"+contrast+".svg")

        average_dice_diff = sum(diff.values())/len(diff)
        print "Average dice difference : "+str(average_dice_diff)

    except Exception, e:
        print e

def root_mean_square_error(a):
    from numpy import sqrt as root
    from numpy import mean, square

    return root(mean(square(a)))


def usage():
    print """
    """ + os.path.basename(__file__) + """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Brainhack MTL 2015

    DESCRIPTION
      Validation script for SCAD (Spinal Cord Automatic Detection)

    USAGE
      """ + os.path.basename(__file__) + """ <algorithm_name>

    MANDATORY ARGUMENTS
      <algorithm_name>  name of the script you want to validate. The script must have -i, -o and -v options enabled.

    OPTIONAL ARGUMENTS
      scad          Checks the result of scad for a data set and outputs the average, rmse, std and max error
      propseg       Verifies the out put of scad and of
      -h                help. Show this message
    """
    sys.exit(1)

# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # reading the name of algorithm from arguments
    script_arguments = sys.argv[1:]
    if "-h" in script_arguments:
        usage()

    ### Start of not good code
    if "-scad" in script_arguments:
        folder = script_arguments[script_arguments.index("-i") + 1]
        contrast = script_arguments[script_arguments.index("-t") + 1]
        if folder != "" or folder is not None:
            validate_scad(folder, contrast)

    if "-propseg" in script_arguments:
        folder = script_arguments[script_arguments.index("-i") + 1]
        contrast = script_arguments[script_arguments.index("-t") + 1]
        if folder != "" or folder is not None:
            scad_propseg_validation(folder, contrast)

    if "-check_dices" in script_arguments:
        folder = script_arguments[script_arguments.index("-i") + 1]
        contrast = script_arguments[script_arguments.index("-t") + 1]
        if folder != "" or folder is not None:
            check_dices(folder, contrast)
    # elif len(script_arguments) > 3:
    #     print 'ERROR: this script only accepts three arguments: the name of your algorithm, if it is a python script or' \
    #           'not and the verbose option.'
    #     usage()
    #
    # algorithm = script_arguments[0]
    # verbose = True
    # ispython = False
    # if len(script_arguments) >= 2:
    #     if 'verbose' in script_arguments[1:]:
    #         verbose = False
    #     if 'ispython' in script_arguments[1:]:
    #         ispython = True
    #
    # scadMRValidation(algorithm=algorithm, isPython=ispython, verbose=verbose)
