


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
from sct_label_vertebrae import  main as sct_label_verterbrae
from sct_register_to_template import main as sct_register_to_template
from sct_label_utils import main as sct_label_utils
from sct_apply_transfo import main as sct_apply_transfo
from sct_maths import main as sct_maths
import glob

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
                sct_deepseg_sc(['-i', os.path.join(root, filename), '-c', contrast, '-ofolder', temp])
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
            sct_label_verterbrae(['-i', file_input, '-s', file_seg_input, '-c', contrast_label, '-ofolder', temp])
            filename_label = filename.split(".nii")[0] + "_seg_labeled.nii" + filename.split(".nii")[1]
            filelabel_input = os.path.join(temp, filename_label)

            filename_label_vert = filename_label.split("_seg_labeled.nii")[0] + "_seg_labeled_vert.nii" + filename_label.split(".nii")[1]
            filelabelvert_input = os.path.join(temp, filename_label_vert)

            os.chdir(temp)
            sct_label_utils(['-i', filelabel_input, '-vert-body', '1, 99', '-o', filename_label_vert])
            os.chdir(cwd)

            # HOGancestor registration
            sct_register_to_template(['-i', file_input, '-s', file_seg_input, '-l', filelabelvert_input, '-c', contrast, '-ofolder', temp, '-param', "step=1,type=im_seg,algo=centermassrot,poly=0,slicewise=0"])
            sct_apply_transfo(['-i', file_seg_input, '-d', file_template_seg, '-w', temp + "/warp_anat2template.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatHOG.nii.gz"])
            sct_maths(['-i', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatHOG.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatHOG_bin.nii.gz", '-bin', '0.5'])
            sct.run("sct_dice_coefficient -i " + path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatHOG_bin.nii.gz -d " + file_template_seg + " -bzmax 1 -o " + path_output + "/" + filename.split(".nii")[0] + "HOG.txt")

            # PCA registration
            sct_register_to_template(['-i', file_input, '-s', file_seg_input, '-l', filelabelvert_input, '-c', contrast, '-ofolder', temp, '-param', "step=1,type=seg,algo=centermassrot,rot=1,poly=0,slicewise=0"])
            sct_apply_transfo(['-i', file_seg_input, '-d', file_template_seg, '-w', temp + "/warp_anat2template.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatPCA.nii.gz"])
            sct_maths(['-i', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatPCA.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatPCA_bin.nii.gz", '-bin', '0.5'])
            sct.run("sct_dice_coefficient -i " + path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatPCA_bin.nii.gz -d " + file_template_seg + " -bzmax 1 -o " + path_output + "/" + filename.split(".nii")[0] + "PCA.txt")

            # No rotation registration
            sct_register_to_template(['-i', file_input, '-s', file_seg_input, '-l', filelabelvert_input, '-c', contrast, '-ofolder', temp, '-param', "step=1,type=seg,algo=centermass,slicewise=0"])
            sct_apply_transfo(['-i', file_seg_input, '-d', file_template_seg, '-w', temp + "/warp_anat2template.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatNoRot.nii.gz"])
            sct_maths(['-i', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatNoRot.nii.gz", '-o', path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatNoRot_bin.nii.gz", '-bin', '0.5'])
            sct.run("sct_dice_coefficient -i " + path_output + "/" + filename.split(".nii")[0] + "seg_reg2anatNoRot_bin.nii.gz -d " + file_template_seg + " -bzmax 1 -o " + path_output + "/" + filename.split(".nii")[0] + "NoRot.txt")

            sct.rmtree(temp, verbose=0)




if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
