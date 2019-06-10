# Python only file
# Load a bunch of simple images, detect axis of symetry with different methods, outputs plots

import sct_utils as sct
import sys, os
import fnmatch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from nicolas_scripts.functions_sym_rot import *
from scipy import misc

def main(args=None):

    if args is None:
        args = sys.argv[1:]

    image_directory = args[0]
    list_image_fname = list(set(os.listdir(image_directory)) - set(fnmatch.filter(os.listdir(image_directory), "*sym_*")))  # list files without "sym_" in it


    for image_fname in list_image_fname:

        image = misc.imread(image_directory + "/" + image_fname).astype(int)

        if len(image.shape) > 2:
            image = np.mean(image[:, :, 0:2], axis=2)

        image_axes = np.zeros((image.shape[0], image.shape[1], 3))
        image_axes[:, :, 0] = image

        seg_image = (image > np.mean(np.concatenate(image))).astype(int)

        angle_hog, conf_score, centermass = find_angle(image, seg_image, 1, 1, "hog", angle_range=90, return_centermass=True, save_figure_path=image_directory + "/fig_sym_" + image_fname)
        if angle_hog is None:
            angle_hog = 0
        angle_pca, _, centermass = find_angle(image, seg_image, 1, 1, "pca", angle_range=90, return_centermass=True)
        image_axes[:, :, 1] = generate_2Dimage_line(image, centermass[0], centermass[1], angle_hog)
        image_axes[:, :, 2] = generate_2Dimage_line(image, centermass[0], centermass[1], angle_pca)

        misc.imsave(image_directory + "/sym_" + image_fname.split(".")[0] + ".png", image_axes, "png")


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
        matplotlib.use("Agg")
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
