# This is script will be moved elsewhere (not supposed to be present in sct)
# test symmetry detection method on sample images

import os
import numpy as np
from spinalcordtoolbox.image import Image
import matplotlib.pyplot as plt
from msct_parser import Parser
import sct_utils as sct
import sys, os, shutil

from sct_axial_rotation import symmetry_angle, save_nifti_like, generate_2Dimage_line


def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Blablablabla')
    parser.add_option(name="-i",
                      type_value="folder",
                      description="Folder to test images",
                      mandatory=True,
                      example="/home/Documents")
    parser.add_option(name="-o",
                      type_value="folder",
                      description="output folder",
                      mandatory=True,
                      example="path/to/output")
    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    arguments = parser.parse(args)
    folder_test = arguments['-i']
    folder_output = arguments['-o']
    fname_test = []
    data_test = []

    fname_test.append(folder_test + "/sub-01/anat/sub-01_T2star.nii.gz")
    data_test.append(np.array(Image(fname_test[0]).data[:, :, 7]))

    # fname_test.append(folder_test + "/sub-02/anat/sub-02_T2star.nii.gz")
    # data_test.append(np.array(Image(fname_test[0]).data[:, :, 7]))
    #
    # fname_test.append(folder_test + "/sub-02/anat/sub-02_T2star.nii.gz")
    # data_test.append(np.array(Image(fname_test[0]).data[:, :, 7]))


    nb_axes = 6

    kmedian_size = 3

    for no, file in enumerate(fname_test):

        angles = symmetry_angle(data_test[0], nb_axes=nb_axes, kmedian_size=kmedian_size, figure=False)
        # centermass = image[0].mean(1).round().astype(int)  # will act weird if image is non binary
        centermass = [int(round(data_test[0].shape[0]/2)), int(round(data_test[0].shape[1]/2))]  # center of image

        image_wline = data_test[0]

        for i_angle in range(0, len(angles)):

            image_wline = generate_2Dimage_line(image_wline, centermass[0], centermass[1], angles[i_angle]-135)

        save_nifti_like(data=image_wline, fname="test_sym" + str(no) + ".nii", fname_like=file, ofolder=folder_output)




if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()