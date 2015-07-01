#!/usr/bin/env python

import os, sys, commands

# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


os.chdir('/Users/tamag/Desktop/GM_atlas/def_new_atlas')
# img_1 = mpimg.imread('gm_white_inv.png')
# img = mpimg.imread('greyscale_select.png')

#grey = rgb2gray(img)

fname = 'greyscale_select_smooth.png'
image = Image.open(fname).convert("L")
arr = np.asarray(image)
print arr[550,550]
#arr_bin_2 = np.zeros((arr.shape[0]-98, arr.shape[1]-95), dtype=np.uint8)
arr_bin = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if 200 > arr[i,j] > 20:
            arr_bin[i,j] = 255
        else: arr_bin[i,j] = 0

# for i in range(arr_bin_2.shape[0]):
#     for j in range(arr_bin_2.shape[1]):
#         arr_bin_2[i,j] = arr[i+7, j+20]

#arr_bin = arr[:, :] > 20
im = Image.fromarray(arr_bin)
im.save('gm_white.png')
# #
# #
# # plt.imshow(arr_bin, cmap=cm.binary)
# # plt.show()
# plt.imshow(arr_bin, cmap=cm.binary)
# plt.show()


# from scipy.ndimage import gaussian_filter, median_filter
#
#
# #kernel = np.ones((5,5),np.float32)/25
# #img_smooth_1 = gaussian_filter(img, sigma=(20, 20), order=0)
# img_smooth_2 = median_filter(image, size=(30,30))
# img_smooth_2.astype(dtype='uint8')
#
# im = Image.fromarray(img_smooth_2)
# #im_1 = Image.fromarray(img_1)
# if im.mode != 'RGB':
#     im2 = im.convert('RGB')
# im2.save('gm_white_inv_smooth.png')
#
# plt.subplot(2,1,1)
# plt.imshow(image, cmap=cm.binary)
# # plt.subplot(2,2,2)
# # plt.imshow(img_smooth_1, cmap=cm.binary)
# plt.subplot(2,1,2)
# plt.imshow(img_smooth_2, cmap=cm.binary)
# plt.show()



bla