#!/usr/bin/env python

import os, sys, commands

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, 'scripts'))


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave
from copy import copy


path = os.path.join(path_sct, "GM_atlas", "raw_data", "test")
fname1 = 'greyscale_final_resampled_registered_crop_resized.png'
fname2 = 'atlas_grays_cerv_sym_correc_r5.png'
fname3 = 'greyscale_reg_no_trans.png'
fname4 = 'greyscale_reg_no_trans_sym.png'
# input: (images must be oriented horizontally with the image viewer)
#   -atlas_grays_cerv_sym_correc_r5.png
#   -greyscale_final_resampled_registered_crop_resized.png
# output:
#   -concatenation.png


def main():

    os.chdir(path)

    ##process:


    #anti_trans: input: greyscale_final_resampled_registered_crop_resized.png    output: greyscale_reg_no_trans.png (old: greyscale_final_reg_no_trans.png)
    print '\nReplacing transition pixel between zones...'
    anti_trans(fname1, list=[0,44,80,120,150,190,220,255], name_output=fname3)

    #copy left side on right side: input: greyscale_reg_no_trans.png  output: greyscale_reg_no_trans_sym.png
    print '\nCopying left side of the image on the right side with change of values...'
    antisym_im(fname3, name_output=fname4)


    #concatenation of GM and WM tracts: inputs: atlas_grays_cerv_sym_correc_r5.png  and  greyscale_reg_no_trans_sym.png  output: concatenation.png
    print '\nConcatenating WM and GM tracts...'
    concatenate_WM_and_GM(WM_file=fname2, GM_file=fname4, name_output='concatenation.png')

    #small hand corrections:  input: concatenation.png  output: concatenation_corrected.png



# avoid transition between zones
def anti_trans(fname, list=[0,45,100,170,255],name_output='notrans.png'):
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


# Concatenate WM and GM tracts
def concatenate_WM_and_GM(WM_file, GM_file, name_output='concatenation.png'):
    #Open file
    im_1 = Image.open(WM_file).convert("L")
    im_2 = Image.open(GM_file).convert("L")

    # Take array
    arr_1 = np.asarray(im_1)
    arr_2 = np.asarray(im_2)
    arr_3 = np.zeros((arr_1.shape[0], arr_1.shape[1]), dtype=np.uint8)
    arr_4 = np.zeros((arr_1.shape[0], arr_1.shape[1]), dtype=np.uint8)

    # Set GM area of WM_file to zero
    for i in range(arr_1.shape[0]):
        for j in range(arr_1.shape[1]):
            if arr_1[i,j] < 235 or arr_1[i,j]==255:
                arr_4[i,j] = arr_1[i,j]
            else: arr_4[i,j] = 0
    im_4 = Image.fromarray(arr_4)
    im_4.save('WM_file_GM_to_0.png')

    # Set WM area of GM_file to zero
    for i in range(arr_1.shape[0]):
        for j in range(arr_2.shape[1]):
            if arr_2[i,j] < 240:
                arr_3[i,j] = arr_2[i,j]
            else: arr_3[i,j] = 0
    im_3 = Image.fromarray(arr_3)
    im_3.save('GM_file_WM_to_zero.png')

    # Concatenate the two files
    arr_o = copy(arr_4)
    for i in range(arr_1.shape[0]):
        for j in range(arr_1.shape[1]):
            if arr_4[i,j] == 0:
                arr_o[i,j] = arr_2[i,j]
    im_o = Image.fromarray(arr_o)
    im_o.save(name_output)

# Make an antisymetric image
def antisym_im(fname, name_output='antisym.png'):
    im_i = Image.open(fname).convert("L")
    arr = np.asarray(im_i)
    arr_bin = copy(arr)
    middle_y = int(round(arr.shape[1]/2.0))
    for i in range(arr.shape[0]):
        for j in range(0,middle_y):
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

