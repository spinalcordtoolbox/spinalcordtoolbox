#!/usr/bin/env python

# Script used to process one MRI image at a time (the image and its segmentation), made to be used with wrapper or alone

import sys, os
import sct_utils as sct
from msct_parser import Parser
from sct_register_to_template import main as sct_register_to_template
from sct_label_vertebrae import main as sct_label_vertebrae
from sct_apply_transfo import main as sct_apply_transfo
from sct_label_utils import main as sct_labels_utils
from sct_maths import main as sct_maths
from nicolas_scripts.functions_sym_rot import *
import csv

def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Script to process a MRI image with its segmentation, blablabla what does this script do')
    parser.add_option(name="-i",
                      type_value="file",
                      description="File input",
                      mandatory=True,
                      example="/home/data/cool_T2_MRI.nii.gz")
    parser.add_option(name="-iseg",
                      type_value="file",
                      description="Segmentation of the input file",
                      mandatory=True,
                      example="/home/data/cool_T2_MRI_seg_manual.nii.gz")
    parser.add_option(name="-o",
                      type_value="folder",
                      description="output folder for test results",
                      mandatory=False,
                      example="path/to/output/folder")

    return parser

def main(args=None):

    #TODO define filenames

    # Parser :
    if not args:
        args = sys.argv[1:]
    parser = get_parser()
    arguments = parser.parse(args)
    fname_image = arguments['-i']
    fname_seg = arguments['-iseg']
    output_dir = arguments['-o']

    # creating output dir if it does not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    fname_seg_template = os.path.join(sct.__data_dir__, 'PAM50/template/PAM50_cord.nii.gz')

    sct.printv("        Python processing file : " + fname_image + " with seg : " + fname_seg)

    # Determining contrast :
    if ("T1w" in fname_image) or ("t1w" in fname_image):
        contrast, contrast_label = "t1", "t1"
    elif ("T2w" in fname_image) or ("t2w" in fname_image):
        contrast, contrast_label = "t2", "t2"
    elif ("T2s" in fname_image) or ("t2s" in fname_image):
        contrast, contrast_label = "t2s", "t2"
    else:
        sct.printv("Contrast not supported yet for file : " + fname_image)
        return

    # Labelling vertebrae :
    sct_label_vertebrae(['-i', fname_image, '-s', fname_seg, '-c', contrast_label, '-ofolder', output_dir, '-v', '1'])
    label_max = np.max(Image(output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_labeled.nii.gz").data)
    sct_labels_utils(['-i', output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_labeled.nii.gz", '-vert-body', "1," + str(int(label_max)), '-o', output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_indiv_labels.nii.gz"])

    # Applying same process but for different methods :

    for method in ["NoRot", "pca", "pca"]:

        sct.printv("\n\n Registration with " + method)

        # Registration
        if method == "NoRot":
            sct_register_to_template(
                ['-i', fname_image, '-s', fname_seg, '-c', contrast, '-l',
                 output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_indiv_labels.nii.gz", '-ofolder',
                 output_dir, '-param',
                 "step=1,type=seg,algo=centermass,poly=0,slicewise=1", '-v', '0'])
        else:
            sct_register_to_template(
                ['-i', fname_image, '-s', fname_seg, '-c', contrast, '-l',
                 output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_indiv_labels.nii.gz", '-ofolder', output_dir, '-param',
                 "step=1,type=seg,algo=centermassrot,poly=0,slicewise=1,rot_method=" + method, '-v', '0'])

        # Applying warping field to segmentation
        sct_apply_transfo(['-i', fname_seg, '-d', fname_seg_template, '-w', output_dir + "/warp_anat2template.nii.gz", '-o', output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_reg.nii.gz"])
        sct_maths(['-i', output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_reg.nii.gz", '-bin', '0.5', '-o', output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_reg_tresh.nii.gz"])

        # Opening registered segmentation
        data_seg_reg = Image(output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_reg_tresh.nii.gz").data
        data_seg_template = Image(fname_seg_template).data
        min_z = np.min(np.nonzero(data_seg_reg)[2])
        max_z = np.max(np.nonzero(data_seg_reg)[2])

        # Computing Dice metrics
        dice_slice = []
        dice_glob = compute_similarity_metric(data_seg_reg[:, :, min_z:max_z], data_seg_template[:, :, min_z:max_z], metric="Dice")

        for z in range(min_z, max_z):
            dice_slice.append(compute_similarity_metric(data_seg_reg[:, :, z], data_seg_template[:, :, z], metric="Dice"))

        # Writing out metrics in csv files
        cwd = os.getcwd()
        os.chdir(output_dir)
        with open((fname_image.split("/")[-1]).split(".nii")[0] + "_dice_" + method + ".csv", 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["dice_global", dice_glob])
            filewriter.writerow(["dice_mean", np.mean(dice_slice)])
            filewriter.writerow(["dice_min", min(dice_slice)])
            filewriter.writerow(["dice_max", max(dice_slice)])
            filewriter.writerow(["dice_std", np.std(dice_slice)])
        os.chdir(cwd)


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
