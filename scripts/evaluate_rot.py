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
from spinalcordtoolbox.reports.qc import generate_qc
import csv
import time
import math

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
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      mandatory=False)

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
    if '-qc' in arguments:
        path_qc = arguments['-qc']
        # creating qc dir if it does not exist
        if not os.path.isdir(path_qc):
            os.mkdir(path_qc)
    if '-o' in arguments:
        output_dir = arguments['-o']
    else:
        output_dir = os.getcwd()

    sct.printv("======> Python processing file : " + fname_image + " with seg : " + fname_seg)

    sub_and_sequence = (fname_image.split("/")[-1]).split(".nii.gz")[0]

    data_image = Image(fname_image).data
    data_seg = Image(fname_seg).data

    nx, ny, nz, nt, px, py, pz, pt = Image(fname_image).dim

    min_z = np.min(np.nonzero(data_seg)[2])
    max_z = np.max(np.nonzero(data_seg)[2])

    angles = np.zeros(max_z - min_z)
    proba_map = np.zeros((nx, ny, nz))

    methods = ["pca", "hog"]

    angle_range = 90

    for k, method in enumerate(methods):

        axes_image = np.zeros((nx, ny, nz))
        start_time = time.time()
        for z in range(0, max_z-min_z):

            if method is "hog":
                angles[z], centermass, proba_map[:, :, z] = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z], px, py, method, angle_range=angle_range, return_centermass=True, return_proba_map=True)
                if math.isnan(angles[z]):
                    sct.printv("confidence score bellow threshold (or not found)")
                    angles[z] = 0
            else:
                angles[z], centermass = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z],px, py, method, angle_range=angle_range, return_centermass=True, return_proba_map=False)

            axes_image[:, :, min_z + z] = generate_2Dimage_line(axes_image[:, :, min_z + z], centermass[0], centermass[1], angles[z]-pi/2, value=k+1)

        sct.printv("Time elapsed for method " + method + " (+ generating axes) : " + str(round(time.time() - start_time, 1)) + " seconds")
        sct.printv("Max angle is : " + str(max(angles) * 180/pi) + ", min is : " + str(min(angles) * 180/pi) + " and mean is : " + str(np.mean(angles) * 180/pi))

        fname_axes = output_dir + "/" + sub_and_sequence + "_axes_" + method + ".nii.gz"
        Image(axes_image, hdr=Image(fname_seg).hdr).save(fname_axes)
        if method is "hog":
            Image(proba_map, hdr=Image(fname_seg).hdr).save(output_dir + "/" + sub_and_sequence + "_hog_proba_map.nii.gz")

        # generate_qc(fname_in1=fname_image, fname_in2=fname_axes, fname_seg=None, args=None, path_qc=path_qc, dataset=None, subject=None, process="rotation")

        # fsleyes /home/nicolas/unf_test/unf_spineGeneric/sub-01/anat/sub-01_T1w.nii.gz /home/nicolas/test_single_rot/sub-01_T1w_axes_pca.nii.gz -cm blue /home/nicolas/test_single_rot/sub-01_T1w_axes_hog.nii.gz -cm green

def memory_limit():
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

if __name__ == '__main__':

    if sys.gettrace() is None:
        sct.init_sct()
        # call main function
        main()
    else:
        memory_limit()  # Limitates maximun memory usage to half
        try:
            sct.init_sct()
            # call main function
            main()
        except MemoryError:
            sys.stderr.write('\n\nERROR: Memory Exception\n')
            sys.exit(1)
