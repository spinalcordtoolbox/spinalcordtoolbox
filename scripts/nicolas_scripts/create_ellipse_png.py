import sct_utils as sct
import sys, os
import fnmatch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from nicolas_scripts.functions_sym_rot import *
from scipy import misc
import skimage.draw as draw
from skimage.filters import gaussian


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    path_output = args[0]

    for ratio in [0.6, 0.7, 0.8, 0.9]:
        for rotation in [-10, -8, -4, -2, 0, 1, 3, 5, 7, 10]:
            for sigma in [0, 1, 2, 4, 8]:

                create_ellipse(path_output, dim=256, ratio=ratio, rotation=rotation, sigma=sigma)





def create_ellipse(path_output, dim, ratio, rotation, sigma=4):

    shape = (dim, dim)
    shape_full = (dim, dim, 3)
    image = np.zeros(shape)
    image_full = np.zeros(shape_full)

    a = dim//4
    b = a * ratio

    name = "ElliTest_ratio_" + str(round(ratio*10)) + "_rot_" + str(rotation) + "_sigma_" + str(round(sigma))

    coordx, coordy = draw.ellipse(dim//2, dim//2, dim//3, dim//3, rotation=0 * pi/180, shape=shape)

    image[coordx, coordy] = 0.5

    coordx, coordy = draw.ellipse(dim//2, dim//2, b, a, rotation=rotation * pi/180, shape=shape)

    image[coordx, coordy] = 1

    image = gaussian_filter(image, sigma=sigma)

    image_full[:, :, 0] = image
    image_full[:, :, 1] = image
    image_full[:, :, 2] = image

    misc.imsave(path_output + "/" + name + ".png", image_full, "png")


if __name__ == '__main__':

    main()
