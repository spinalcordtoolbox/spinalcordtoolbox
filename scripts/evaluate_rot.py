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

    data_image = Image(fname_image).data
    data_seg = Image(fname_seg).data

    nx, ny, nz, nt, px, py, pz, pt = Image(fname_image).dim

    min_z = np.min(np.nonzero(data_seg)[2])
    max_z = np.max(np.nonzero(data_seg)[2])

    angles = np.zeros(max_z - min_z)

    methods = ["pca", "hog"]

    for k, method in enumerate(methods):

        axes_image = np.zeros((nx, ny, nz))
        for z in range(0, max_z-min_z):

            angles[z], centermass = find_angle(data_image, data_seg, px, py, method, return_centermass=True)
            axes_image[:, :, min_z + z] = generate_2Dimage_line(axes_image[:, :, min_z + z], centermass[0], centermass[1], angles[z], value=k+1)

        name_image_axes = (fname_image.split("/")[-1]).split(".nii.gz")[0] + "_axes_" + method + ".nii.gz"
        fname_axes = output_dir + "/" + name_image_axes
        Image(axes_image, hdr=Image(fname_seg).hdr).save(fname_axes)

        generate_qc(fname_in1=fname_image, fname_in2=fname_axes, fname_seg=None, args=None, path_qc=path_qc, dataset=None, subject=None, process="rotation")


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
