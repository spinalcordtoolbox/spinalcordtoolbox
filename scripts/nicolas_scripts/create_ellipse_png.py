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
import numpy as np


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    path_output = args[0]

    ratio_list = [0.7]
    rotation_list = np.arange(-45, 45 + 1, 1)
    sigma_list = [1]
    dim_list = [256]

    for ratio in ratio_list:
        for rotation in rotation_list:
            for sigma in sigma_list:
                for dim in dim_list:

                    create_ellipse(path_output, dim=dim, ratio=ratio, rotation=rotation, sigma=sigma)





def create_ellipse(path_output, dim, ratio, rotation, sigma=4):

    shape = (dim, dim)
    shape_full = (dim, dim, 3)
    image = np.zeros(shape)
    image_full = np.zeros(shape_full)

    a = dim//4
    b = a * ratio

    name = "ElliTest_ratio_" + str(round(ratio*10)) + "_rot_" + str(rotation) + "_sigma_" + str(round(sigma)) + "_dim_" + str(dim)

    coordx, coordy = draw.ellipse(dim//2, dim//2, dim//3, dim//3, rotation=0 * pi/180, shape=shape)

    image[coordx, coordy] = 0.5

    coordx, coordy = draw.ellipse(dim//2, dim//2, b, a, rotation=rotation * pi/180, shape=shape)

    image[coordx, coordy] = 0.7

    coordx, coordy = draw.ellipse(dim//2, dim//2, b/3*2, a/3*2, rotation=rotation * pi/180, shape=shape)

    image[coordx, coordy] = 0.85

    coordx, coordy = draw.ellipse(dim//2, dim//2, b/3, a/3, rotation=rotation * pi/180, shape=shape)

    image[coordx, coordy] = 1

    image = gaussian_filter(image, sigma=sigma)

    image_full[:, :, 0] = image
    image_full[:, :, 1] = image
    image_full[:, :, 2] = image

    misc.imsave(path_output + "/" + name + ".png", image_full, "png")


if __name__ == '__main__':

    main()
