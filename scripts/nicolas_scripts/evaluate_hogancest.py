


import os
import numpy as np
from spinalcordtoolbox.image import Image
import matplotlib.pyplot as plt
from msct_parser import Parser
import sct_utils as sct
import sys, os, shutil
from functions_sym_rot import *
import fnmatch
import scipy
from msct_register import compute_pca, angle_between, register2d_centermassrot
from sct_deepseg_sc import main as sct_deepseg_sc
from sct_label_vertebrae import  main as sct_label_vertebrae
from sct_register_to_template import main as sct_register_to_template
from sct_label_utils import main as sct_label_utils
from sct_apply_transfo import main as sct_apply_transfo
from sct_maths import main as sct_maths
import glob
import csv
import matplotlib.pyplot as plt
import matplotlib

def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Blablablabla')
    parser.add_option(name="-i",
                      type_value="folder",
                      description="Input folder with data used for the test",
                      mandatory=True,
                      example="/home/data")

    parser.add_option(name="-test",  # TODO find better name
                      type_value="str",
                      description="put name of the test you want to run",
                      mandatory=True,
                      example="src_seg.nii.gz")
    parser.add_option(name="-o",
                      type_value="folder",
                      description="output folder for test results",
                      mandatory=False,
                      example="path/to/output/folder")

    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    cwd = os.getcwd()

    parser = get_parser()
    arguments = parser.parse(args)
    input_folder = arguments['-i']
    test_str = arguments['-test']
    if '-o' in arguments:
        path_output = arguments['-o']
    else:
        path_output = os.getcwd()

    file_template_seg = path_output + "/template_seg.nii.gz"
    sct.copy(sct.__data_dir__ + "/PAM50/template/PAM50_cord.nii.gz", file_template_seg, verbose=0)  # copy in output for latter comparison

    dice_coeff_HOG = []
    dice_coeff_PCA = []
    dice_coeff_NoRot = []

    for root, dirnames, filenames in os.walk(input_folder):  # searching the given directory

        # search the input dir for MRI files
        for filename in fnmatch.filter(filenames, "*.nii*"):  # if file with nii extension (.nii or .nii.gz) found
            if "seg" in filename:
                continue  # do not consider it if it's a segmentation
            if "dwi" in filename:
                continue  # do not consider it if it's DWI
            if "MT" in filename:
                continue  # do not consider it if it's MT

            # create temp directory deleted at the end of the loop
            temp = path_output + "/temp"
            if not os.path.exists(temp):
                os.makedirs(temp)

            filename_seg = filename.split(".nii")[0] + "_seg.nii" + filename.split(".nii")[1]

            # determine contrast of the file
            if ("T1w" in filename) or ("t1w" in filename):
                contrast = "t1"
            elif ("T2w" in filename) or ("t2w" in filename):
                contrast = "t2"
            elif ("T2s" in filename) or ("t2s" in filename):
                contrast = "t2s"
            else:
                sct.printv("could not find contrast for file : " + filename)
                continue

            if filename_seg in filenames:  # find potential seg associated with the file
                # ok segmentation exists
                sct.copy(os.path.join(root, filename_seg), temp + "/" + filename_seg, verbose=0)  # copy to temp
            else:
                sct_deepseg_sc(['-i', os.path.join(root, filename), '-c', contrast, '-ofolder', temp, '-v', '0'])
                # seg output to temp

            # sct.copy(os.path.join(temp, filename_seg), path_output + "/" + filename_seg, verbose=0)  # copy in output for latter comparison
            # TODO maybe compare seg anat with seg template registered to anat

            file_seg_input = os.path.join(root, filename_seg)  # name and path of fileseg
            file_input = os.path.join(root, filename)
            sct.copy(file_input, temp + "/" + filename, verbose=0)  # copy original file

            if contrast == "t2s":
                contrast_label = "t2"  # label only has t1 or t2 has contrast input
            else:
                contrast_label = contrast
            # label the vertebrae
            sct_label_vertebrae(['-i', file_input, '-s', file_seg_input, '-c', contrast_label, '-ofolder', temp, '-v', '0'])
            filename_label = filename.split(".nii")[0] + "_seg_labeled.nii" + filename.split(".nii")[1]
            filelabel_input = os.path.join(temp, filename_label)

            filename_label_vert = filename_label.split("_seg_labeled.nii")[0] + "_seg_labeled_vert.nii" + filename_label.split(".nii")[1]
            filelabelvert_input = os.path.join(temp, filename_label_vert)

            os.chdir(temp)
            sct_label_utils(['-i', filelabel_input, '-vert-body', '1, 99', '-o', filename_label_vert, '-v', '0'])
            os.chdir(cwd)

            # HOGancestor registration
            sct_register_to_template(['-i', file_input, '-s', file_seg_input, '-l', filelabelvert_input, '-c', contrast, '-ofolder', temp, '-param', "step=1,type=im,algo=centermassrot,poly=0,slicewise=0,rot_method=HOG", '-v', '0'])
            sct_apply_transfo(['-i', file_seg_input, '-d', file_template_seg, '-w', temp + "/warp_anat2template.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatHOG.nii.gz", '-v', '0'])
            sct_maths(['-i', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatHOG.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatHOG_bin.nii.gz", '-bin', '0.5', '-v', '0'])
            seg_anat_reg = Image(path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatHOG_bin.nii.gz").data
            min_z = np.min(np.nonzero(seg_anat_reg)[2])
            max_z = np.max(np.nonzero(seg_anat_reg)[2]) + 1  # python indexing
            seg_temp = Image(file_template_seg).data
            dice_coeff_HOG.append(compute_similarity_metric(seg_anat_reg[:, :, min_z:max_z], seg_temp[:, :, min_z:max_z]))

            # PCA registration
            sct_register_to_template(['-i', file_input, '-s', file_seg_input, '-l', filelabelvert_input, '-c', contrast, '-ofolder', temp, '-param', "step=1,type=seg,algo=centermassrot,rot=1,poly=0,slicewise=0", '-v', '0'])
            sct_apply_transfo(['-i', file_seg_input, '-d', file_template_seg, '-w', temp + "/warp_anat2template.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatPCA.nii.gz", '-v', '0'])
            sct_maths(['-i', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatPCA.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatPCA_bin.nii.gz", '-bin', '0.5', '-v', '0'])
            seg_anat_reg = Image(path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatPCA_bin.nii.gz").data
            min_z = np.min(np.nonzero(seg_anat_reg)[2])
            max_z = np.max(np.nonzero(seg_anat_reg)[2]) + 1  # python indexing
            seg_temp = Image(file_template_seg).data
            dice_coeff_PCA.append(compute_similarity_metric(seg_anat_reg[:, :, min_z:max_z], seg_temp[:, :, min_z:max_z]))

            # No rotation registration
            sct_register_to_template(['-i', file_input, '-s', file_seg_input, '-l', filelabelvert_input, '-c', contrast, '-ofolder', temp, '-param', "step=1,type=seg,algo=centermass,slicewise=0", '-v', '0'])
            sct_apply_transfo(['-i', file_seg_input, '-d', file_template_seg, '-w', temp + "/warp_anat2template.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatNoRot.nii.gz", '-v', '0'])
            sct_maths(['-i', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatNoRot.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatNoRot_bin.nii.gz", '-bin', '0.5', '-v', '0'])
            seg_anat_reg = Image(path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatNoRot_bin.nii.gz").data
            min_z = np.min(np.nonzero(seg_anat_reg)[2])
            max_z = np.max(np.nonzero(seg_anat_reg)[2]) + 1  # python indexing
            seg_temp = Image(file_template_seg).data
            dice_coeff_NoRot.append(compute_similarity_metric(seg_anat_reg[:, :, min_z:max_z], seg_temp[:, :, min_z:max_z]))

            sct.rmtree(temp, verbose=0)

    os.chdir(path_output)
    csvData = [['Dice_coeff_HOG'], dice_coeff_HOG, ['Dice_coeff_PCA'], dice_coeff_PCA, ['Dice_coeff_NoRot'], dice_coeff_NoRot]
    with open('Dice_result.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()

    matplotlib.use('Agg')  # prevent display figure
    plt.figure(figsize=(8, 8))
    plt.title("Dice coefficient for different files")
    plt.plot(range(0, len(dice_coeff_HOG)), dice_coeff_HOG, "bo")
    plt.plot(range(0, len(dice_coeff_PCA)), dice_coeff_PCA, "ro")
    plt.plot(range(0, len(dice_coeff_NoRot)), dice_coeff_NoRot, "go")
    plt.legend(("Hogancest", "PCA", "NoRot"))
    plt.xlabel("file number")
    plt.ylabel("Dice coefficient")
    plt.savefig("Dice_coeff.png")
    plt.close()

    os.chdir(cwd)

    sct.printv("Mean Dice coeff for HOG is : " + str(np.mean(dice_coeff_HOG)))
    sct.printv("Mean Dice coeff for PCA is : " + str(np.mean(dice_coeff_PCA)))
    sct.printv("Mean Dice coeff for NoRot is : " + str(np.mean(dice_coeff_NoRot)))




if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
