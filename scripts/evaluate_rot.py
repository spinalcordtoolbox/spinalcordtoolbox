#!/usr/bin/env python

# Script used to process one MRI image at a time (the image and its segmentation), made to be used with wrapper or alone


from __future__ import division, absolute_import
import sys, os
import sct_utils as sct
from msct_parser import Parser
from nicolas_scripts.functions_sym_rot import *
from spinalcordtoolbox.reports.qc import generate_qc
import csv
import time
import math
from scipy.ndimage.filters import gaussian_filter1d

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
    cwd = os.getcwd()
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

    image_object = Image(fname_image).change_orientation("LPI")
    seg_object = Image(fname_seg).change_orientation("LPI")

    fname_image_output = output_dir + "/" + sub_and_sequence + ".nii.gz"
    fname_seg_output = output_dir + "/" + sub_and_sequence + "_seg.nii.gz"

    data_image = image_object.data
    data_seg = seg_object.data

    nx, ny, nz, nt, px, py, pz, pt = seg_object.dim

    min_z = np.min(np.nonzero(data_seg)[2])
    max_z = np.max(np.nonzero(data_seg)[2])

    methods = ["pca", "hog", "auto"]

    angle_range = 20
    conf_score_th_pca = 1.6  # for pca and auto !
    conf_score_th_hog = 1  # only for hog
    smooth = True

    for k, method in enumerate(methods):

        angles = np.zeros(max_z - min_z)
        conf_score = np.zeros(max_z - min_z)
        axes_image = np.zeros((nx, ny, nz))
        start_time = time.time()
        centermass = np.zeros((2, max_z-min_z))

        for z in range(0, max_z-min_z):

            if method is "hog":
                angles[z], conf_score[z], centermass[:, z] = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z], px, py, method, angle_range=angle_range, return_centermass=True)
                if math.isnan(angles[z]) or conf_score[z] is None:
                    raise Exception("this is not supposed to happen, hog is only searching in the angle range, no angle should be outside range")
                if conf_score[z] < conf_score_th_hog:
                    angles[z] = 0
                    conf_score[z] = -5
            elif method is "pca":
                angles[z], conf_score[z], centermass[:, z] = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z], px, py, method, angle_range=angle_range, return_centermass=True)
                if math.isnan(conf_score[z]) or conf_score[z] is None:
                    conf_score[z] = -10
                    angles[z] = 0
                if conf_score[z] < conf_score_th_pca:
                    angles[z] = 0
                    conf_score[z] = -5
            elif method is "auto":
                angles[z], conf_score[z], centermass[:, z] = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z], px, py, "pca", angle_range=angle_range, return_centermass=True)
                if conf_score[z] < conf_score_th_pca or math.isnan(conf_score[z]) or conf_score[z] is None:
                    angles[z], conf_score[z], centermass[:, z] = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z], px, py, "hog", angle_range=angle_range, return_centermass=True)
            else:
                raise Exception("no method named" + method)

        z_nonzero = range(0, max_z-min_z)

        if smooth:
            # coeffs = np.polyfit(z_nonzero, angles[z_nonzero], polydeg)
            # poly = np.poly1d(coeffs)
            # angles_smoothed = np.polyval(poly, z_nonzero)
            angles_smoothed = gaussian_filter1d(angles, 3)

        for z in range(0, max_z-min_z):
            axes_image[:, :, min_z + z] = generate_2Dimage_line(axes_image[:, :, min_z + z], centermass[0, z], centermass[1, z], angles_smoothed[z] - pi/2, value=k+1)
            # axes_image[int(centermass[0]), int(centermass[1]), min_z + z] = 100000

        sct.printv("Time elapsed for method " + method + " (+ generating axes) : " + str(round(time.time() - start_time, 1)) + " seconds")
        sct.printv("Max angle is : " + str(max(angles) * 180/pi) + ", min is : " + str(min(angles) * 180/pi) + " and mean is : " + str(np.mean(angles) * 180/pi))

        fname_axes = output_dir + "/" + sub_and_sequence + "_axes_" + method + ".nii.gz"
        Image(axes_image, hdr=image_object.hdr).save(fname_axes)
        image_object.save(fname_image_output)
        seg_object.save(fname_seg_output)

        # if method is "pca":
        #     cmap = 'PRGn'
        # elif method is "hog":
        #     cmap = 'Wistia'
        # else:
        #     cmap = 'winter'
        # plt.figure(figsize=(6.4 * 2, 4.8 * 2))
        # plt.scatter(z_nonzero, angles * 180 / pi, c=conf_score, cmap=cmap)
        # if smooth:
        #     plt.plot(z_nonzero, angles_smoothed * 180/pi, "r-")
        # plt.ylabel("angle in deg")
        # plt.xlabel("z slice")
        # plt.colorbar().ax.set_ylabel("conf score " + method)
        # plt.savefig(output_dir + "/" + fname_image.split("/")[-1] + "_" + sub_and_sequence + method + "_angle_conf_score_z.png")  # reliable file name ?

        angles_qc = np.zeros(data_image.shape[2])
        angles_qc[:] = np.nan
        angles_qc[min_z:max_z] = -angles_smoothed

        generate_qc(fname_in1=fname_image_output, fname_seg=fname_seg_output, angle_line=angles_qc[::-1], args=[method], path_qc=path_qc, dataset=None, subject=None, process="rotation")

    sct.printv("fsleyes " + fname_image_output + " " + fname_seg_output + " -cm red" + " " + output_dir + "/" + sub_and_sequence + "_axes_pca.nii.gz -cm blue " + output_dir + "/" + sub_and_sequence + "_axes_hog.nii.gz -cm green " + output_dir + "/" + sub_and_sequence + "_axes_auto.nii.gz -cm yellow", type='info')
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

    # if sys.gettrace() is None:
        sct.init_sct()
        # call main function
        main()
    # else:
    #     memory_limit()  # Limitates maximun memory usage to half
    #     try:
    #         sct.init_sct()
    #         call main function
            # main()
        # except MemoryError:
        #     sys.stderr.write('\n\nERROR: Memory Exception\n')
        #     sys.exit(1)
