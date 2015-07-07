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
from copy import copy

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def sorting_value_of_zone(zone):
    group_value=[[[]],[[]]]  # liste(liste_value, liste_nb_of_this-value)
    for i in range(len(zone)):
        if zone[i] not in group_value[0][0]:
            group_value[0][0].append(zone[i])
            group_value[1][0].append(1)
        else:
            index = group_value[0][0].index(zone[i])
            group_value[1][0][index] += 1
    return group_value

os.chdir('/Users/tamag/Desktop/GM_atlas/def_new_atlas/test_registration')

os.chdir('/Users/tamag/Desktop/GM_atlas/def_new_atlas')
# img_1 = mpimg.imread('gm_white_inv.png')
# img = mpimg.imread('greyscale_select.png')

#grey = rgb2gray(img)




# # tests
# list_test = [1,2,3,2,3,3,3,3,3,4]
# output = sorting_value_of_zone(list_test)
# bla

# f = 80
# a = abs(0 - f)
# b = abs(45 - f)
# c = abs(100 - f)
# d = abs(170 - f)
# e = abs(255 - f)
# if a == min(a,b,c,d,e):
#     g = 0
# elif b == min(a,b,c,d,e):
#     g = 45
# elif c == min(a,b,c,d,e):
#     g = 100
# elif d == min(a,b,c,d,e):
#     g = 170
# elif e == min(a,b,c,d,e):
#     g = 255

fname = 'gm_white_resampled_registered.png'
image = Image.open(fname).convert("L")
arr = np.asarray(image)

# avoid transition between zones
a = 0
b = 45
c = 100
d = 170
e = 255
list = [a,b,c,d,e]
#arr_bin_2 = np.zeros((arr.shape[0]-98, arr.shape[1]-95), dtype=np.uint8)
# arr_bin = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
# for i in range(arr.shape[0]):
#     for j in range(arr.shape[1]):
#         if arr[i,j] not in list:
#             # regarder zone 5x5 autour
#             zone = [arr[i+k,j+h] for k in range(-5,5) for h in range(-5,5)]
#             group_value = sorting_value_of_zone(zone)
#             nb_value_searched = max(group_value[1][0])
#             index_value_searched = group_value[1][0].index(nb_value_searched)
#             value_searched = group_value[0][0][index_value_searched]
#             if value_searched == 254:
#                 value_searched +=1
#             if value_searched == 1:
#                 value_searched -=1
#             if value_searched not in list:
#                 print i,j, value_searched
#             arr_bin[i,j] = value_searched
#         else:
#             arr_bin[i,j] = arr[i,j]


arr_bin = np.zeros((451, 613), dtype=np.uint8)
for i in range(451):
    # h = 74+i
    for j in range(613):
        # g = 103+j

        arr_bin[i,j] = arr[74+i,103+j]


# middle_y = int(round(arr.shape[1]/2.0))
# for i in range(arr.shape[0]):
#     for j in range(middle_y,arr.shape[1]):
#         if arr[i,j] == 45:
#             arr_bin[i,j] = 150
#         elif arr[i,j] == 80:
#             arr_bin[i,j] = 190
#         elif arr[i,j] == 120:
#             arr_bin[i,j] = 220
        # else: arr_bin[i,j] = arr[i,j]


#arr_bin = arr[:, :] > 20
im = Image.fromarray(arr_bin)
im.save('gm_white_resampled_registered_crop.png')
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