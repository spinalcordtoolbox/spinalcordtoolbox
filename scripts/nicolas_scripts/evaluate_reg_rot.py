#!/usr/bin/env python

import sys, sct_utils as sct
from msct_parser import Parser
from sct_register_to_template import main as sct_register_to_template
from sct_label_vertebrae import main as sct_label_vertebrae

def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Blablablabla')
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

    # Parser :
    if not args:
        args = sys.argv[1:]
    parser = get_parser()
    arguments = parser.parse(args)
    fname_image = arguments['-i']
    fname_seg = arguments['-iseg']
    output_dir = arguments['-o']

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

    sct_label_vertebrae(['-i', fname_image, '-s', fname_seg, '-c', contrast_label, '-ofolder', output_dir, '-v', '1'])

    sct_register_to_template(
        ['-i', fname_image, '-s', fname_seg, '-c', contrast, '-l',
         output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_labeled.nii.gz", '-ofolder', output_dir, '-param',
         "step=1,type=seg,algo=centermassrot,poly=0,slicewise=1,rot_method=PCA", '-v', '0'])


    #  TODO : write out dice scores as txt ? then func to agregate them

if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
