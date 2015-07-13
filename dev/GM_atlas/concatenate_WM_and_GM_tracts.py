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


path = '/Users/tamag/Desktop/GM_atlas/def_new_atlas/correct_images/results'
fname = 'greyscale_antisym_left_corrected.png'
fname1 = 'greyscale_final.png'
fname2 = 'white_WM_mask_resampled_registered_crop_resized.png'
fname3 = 'white_GM_mask_resampled_registered_crop_resized.png'
fname4 = 'greyscale_final_resampled_registered_crop_resized.png'
fname5 = 'greyscale_final_reg_no_trans_corr.png'
fname6 = 'atlas_grays_cerv_sym_correc_r5.png'
fname7 = 'greyscale_final_for_atlas.png'

def main():

    os.chdir(path)
    # #WM_mask(fname1,name_output='white_WM_mask.png')
    # anti_trans(fname2, list=[0,255], name_output='white_WM_mask_reg_notrans.png')
    # anti_trans(fname3, list=[0,255], name_output='white_GM_mask_reg_notrans.png')
    # anti_trans(fname4, list=[0,44,80,120,150,190,220,255], name_output='greyscale_final_reg_no_trans.png')
    # #correction by hand of greyscale (output:greyscale_final_reg_no_trans_corr.png)
    # antisym_im(fname5,name_output='greyscale_final_for_atlas.png')
    # WM_mask('greyscale_final_for_atlas.png',name_output='WM_mask_for_atlas.png')
    # GM_mask('greyscale_final_for_atlas.png',name_output='GM_mask_for_atlas.png')
    im_1 = Image.open(fname6).convert("L")
    arr_1 = np.asarray(im_1)
    arr_4 = np.zeros((arr_1.shape[0], arr_1.shape[1]), dtype=np.uint8)
    for i in range(arr_1.shape[0]):
        for j in range(arr_1.shape[1]):
            if arr_1[i,j] < 235 or arr_1[i,j]==255:
                arr_4[i,j] = arr_1[i,j]
            else: arr_4[i,j] = 0
    im_4 = Image.fromarray(arr_4)
    im_4.save('b.png')
    im_2 = Image.open(fname7).convert("L")
    arr_2 = np.asarray(im_2)
    arr_3 = np.zeros((arr_1.shape[0], arr_1.shape[1]), dtype=np.uint8)
    for i in range(arr_1.shape[0]):
        for j in range(arr_2.shape[1]):
            if arr_2[i,j] < 240:
                arr_3[i,j] = arr_2[i,j]
            else: arr_3[i,j] = 0
    im_3 = Image.fromarray(arr_3)
    im_3.save('a.png')
    #arr_o = arr_1 + arr_3
    arr_o = copy(arr_4)
    for i in range(arr_1.shape[0]):
        for j in range(arr_1.shape[1]):
            if arr_4[i,j] == 0:
                arr_o[i,j] = arr_2[i,j]
    im_o = Image.fromarray(arr_o)
    im_o.save('addition.png')


# avoid transition between zones
def anti_trans(fname, list=[0,45,100,170,255],name_output='notrans.png'):
    # a = 0
    # b = 45
    # c = 100
    # d = 170
    # e = 255
    # list = [a,b,c,d,e]
    im_i = Image.open(fname).convert("L")
    arr = np.asarray(im_i)
    arr_bin = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i,j] not in list:
                # regarder zone 5x5 autour
                zone = [arr[i+k,j+h] for k in range(-5,5) for h in range(-5,5)]
                group_value = sorting_value_of_zone(zone)
                nb_value_searched = max(group_value[1][0])
                index_value_searched = group_value[1][0].index(nb_value_searched)
                value_searched = group_value[0][0][index_value_searched]
                if value_searched == 254:
                    value_searched +=1
                if value_searched == 1:
                    value_searched -=1
                if value_searched not in list:
                    print i,j, value_searched
                arr_bin[i,j] = value_searched
            else:
                arr_bin[i,j] = arr[i,j]
    im_o = Image.fromarray(arr_bin)
    im_o.save(name_output)


# Make an antisymetric image
def antisym_im(fname, name_output='antisym.png'):
    im_i = Image.open(fname).convert("L")
    arr = np.asarray(im_i)
    arr_bin = copy(arr)
    middle_y = int(round(arr.shape[1]/2.0))
    for i in range(arr.shape[0]):
        for j in range(0,middle_y-1):
            if arr[i,j] == 150:
                arr_bin[i,-j-1] = 45
            elif arr[i,j] == 190:
                arr_bin[i,-j-1] = 80
            elif arr[i,j] == 220:
                arr_bin[i,-j-1] = 120
            else: arr_bin[i,-j-1] = arr[i,j]
    im_o = Image.fromarray(arr_bin)
    im_o.save(name_output)


# Create mask of the grey matter from greyscale image
def GM_mask(fname, value_of_mask=255, name_output='GM_mask.png'):
    im_i = Image.open(fname).convert("L")
    arr = np.asarray(im_i)
    arr_bin = copy(arr)
    for i in range(arr.shape[0]):
        for j in range(0,arr.shape[1]):
            if 40 < arr[i,j] < 225:
                arr_bin[i,j] = 255
            else: arr_bin[i,j] = 0
    im_o = Image.fromarray(arr_bin)
    im_o.save(name_output)

# Create mask of the WM from greyscale image
def WM_mask(fname, value_of_mask=255, name_output='WM_mask.png'):
    im_i = Image.open(fname).convert("L")
    arr = np.asarray(im_i)
    arr_bin = copy(arr)
    for i in range(arr.shape[0]):
        for j in range(0,arr.shape[1]):
            if arr[i,j] == 255:
                arr_bin[i,j] = 255
            else: arr_bin[i,j] = 0
    im_o = Image.fromarray(arr_bin)
    im_o.save(name_output)

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




# #
# #
# # plt.imshow(arr_bin, cmap=cm.binary)
# # plt.show()
# plt.imshow(arr_bin, cmap=cm.binary)
# plt.show()


# from scipy.ndimage import gaussian_filter, median_filter
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



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()

