#!/usr/bin/env python

import os, sys

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, 'scripts'))

import numpy as np
from PIL import Image
import nibabel
import sct_utils as sct
import matplotlib.pyplot as plt
from copy import copy

ext_o = '.nii.gz'
path_info = os.path.join(path_sct, "GM_atlas", "raw_data")
path_output = os.path.join(path_sct, "GM_atlas", "raw_data", "test")

# input: greyscale_final.png
# output: greyscale_final_resampled_registered_crop_resized.png


def main():

    os.chdir(path_output)

    # Define names
    fname1 = 'greyscale_GM_atlas.png'  # greyscale image showing tracts (warp applied to it at the end)
    fname2 = 'white_WM_mask.png'    # white mask of the white matter for registration
    fname3 = 'white_GM_mask.png'           # white mask of the grey matter (warp applied to it at the end)
    fname4 = 'mask_grays_cerv_sym_correc_r5.png' # mask from the WM atlas
    slice = 'slice_ref_template.nii.gz'          #reference slice of the GM template
    name1 = 'greyscale_GM_atlas'
    name2 = 'white_WM_mask'
    name3 = 'white_GM_mask'
    name4 = 'mask_grays_cerv_sym_correc_r5'

    fname10 = 'greyscale_GM_atlas_resampled_registered_crop_resized.png'
    fname11 = 'greyscale_reg_no_trans.png'
    fname12 = 'greyscale_reg_no_trans_sym.png'
    fname13 = 'atlas_grays_cerv_sym_correc_r5_horizontal.png' #horizontal roientation


    # Copy file to path_output
    print('\nCopy files to output folder')
    for file in (
     fname1,
     fname2,
     fname3,
     fname4,
     slice,
     fname13,
     ):
        sct.copy(os.path.join(path_info, file), file)

    print '\nSave nifti images from png'
    save_nii_from_png(fname1)
    save_nii_from_png(fname2)
    save_nii_from_png(fname3)
    save_nii_from_png(fname4)

    # Crop image background

    #sct.run('sct_crop_image -i '+ name2 + ext_o +' -dim 0,1 -start 84,0 -end 1581,1029 -b 0 -o binary_gm_crop.nii.gz')
    #name2 = 'binary_gm_crop'

    # Interpolation of the images to set them in the right format
    print'\nInterpolate images to set them in the right format'
    interpolate(name1+ext_o)
    interpolate(name2+ext_o)
    interpolate(name3+ext_o)

    # Copy some info from hdr of the slice template
    print '\nCopy some info from hdr of the slice template'
    copy_hdr(name1+'_resampled'+ext_o, slice)
    copy_hdr(name2+'_resampled'+ext_o, slice)
    copy_hdr(name3+'_resampled'+ext_o, slice)
    copy_hdr(name4+ext_o, slice)

    # transformation affine pour premier recalage
    print '\nAffine transformation for position registration'
    warp_1_prefix = 'warp_1_'
    tmp_file = 'tmp.nii.gz'
    cmd = ('isct_antsRegistration --dimensionality 2 -m MeanSquares[mask_grays_cerv_sym_correc_r5.nii.gz, '+name2+'_resampled'+ext_o+', 3, 4] -t Affine[50] --convergence 100x10 -s 1x0 -f 4x1 -o '+ warp_1_prefix +' -r [mask_grays_cerv_sym_correc_r5.nii.gz, '+name2+'_resampled'+ext_o+', 0]')
    sct.run(cmd)
    cmd_0 = ('isct_antsApplyTransforms -d 2 -i '+name2+'_resampled'+ext_o+' -t '+warp_1_prefix+'0GenericAffine.mat -r mask_grays_cerv_sym_correc_r5.nii.gz  -o '+ tmp_file)
    sct.run(cmd_0)

    # transformation Syn pour faire correspondre les formes
    print '\nSyn transformation for shape registration'
    warp_2_prefix = 'warp_2_'
    tmp_file_2 = 'tmp_2.nii.gz'
    # cmd_1 = ('isct_antsRegistration --dimensionality 2 -m MeanSquares[mask_grays_cerv_sym_correc_r5.nii.gz, '+ tmp_file+', 1, 4] -t BSplineSyN[0.2,3] --convergence 100x10 -s 1x0 -f 4x1 -o '+warp_2_prefix+' -r [mask_grays_cerv_sym_correc_r5.nii.gz, ' + tmp_file + ', 0]')
    cmd_1 = ('isct_antsRegistration --dimensionality 2 -m MeanSquares[mask_grays_cerv_sym_correc_r5.nii.gz, '+ tmp_file+', 1, 32] -t SyN[0.1,3,0] --convergence 1000x1000 -s 1x0 -f 2x1 -o '+warp_2_prefix+' -r [mask_grays_cerv_sym_correc_r5.nii.gz, ' + tmp_file + ', 0]')
    sct.run(cmd_1)
    cmd_2 = ('isct_antsApplyTransforms -d 2 -i '+tmp_file+' -t '+ warp_2_prefix + '1Warp.nii.gz -r mask_grays_cerv_sym_correc_r5.nii.gz  -o '+tmp_file_2)
    sct.run(cmd_2)

    # Symmetrize the warping fields

    # concatenation et application des warping fields aux images de base
    print'\nConcatenate and apply warping fields to root images'
    # cmd = ('isct_ComposeMultiTransform 2 outwarp.nii.gz  test_registration0GenericAffine.mat test_registration_Bspline1Warp.nii.gz -R mask_grays_cerv_sym_correc_r5.nii.gz')
    # cmd = ('isct_antsApplyTransforms -d 2 -i greyscale_select_inv_resampled.nii.gz -o greyscale_registered.nii.gz -n Linear -t outwarp.nii.gz -r mask_grays_cerv_sym_correc_r5.nii.gz')
    concat_and_apply(inputs=[name1+'_resampled'+ext_o], dest=name4+ext_o,
                     output_names=[name1+'_resampled'+'_registered'+ext_o],
                     warps=[warp_1_prefix+'0GenericAffine.mat', warp_2_prefix + '1Warp.nii.gz'], interpolation='Linear')
    concat_and_apply(inputs=[name2+'_resampled'+ext_o, name3+'_resampled'+ext_o], dest=name4+ext_o,
                 output_names=[name2+'_resampled'+'_registered'+ext_o, name3+'_resampled'+'_registered'+ext_o],
                 warps=[warp_1_prefix+'0GenericAffine.mat', warp_2_prefix + '1Warp.nii.gz'], interpolation='NearestNeighbor')
    # concat_and_apply(inputs=[name1+'_resampled'+ext_o, name2+'_resampled'+ext_o, name3+'_resampled'+ext_o], dest=name4+ext_o,
    #          output_names=[name1+'_resampled'+'_registered'+ext_o, name2+'_resampled'+'_registered'+ext_o, name3+'_resampled'+'_registered'+ext_o],
    #          warps=[warp_1_prefix+'0GenericAffine.mat', warp_2_prefix + '1Warp.nii.gz'], interpolation='Linear')

    # Save png images of the registered NIFTI images
    print '\nSave png images of the registered nifti images'
    save_png_from_nii(name1+'_resampled'+'_registered'+ext_o)
    save_png_from_nii(name2+'_resampled'+'_registered'+ext_o)
    save_png_from_nii(name3+'_resampled'+'_registered'+ext_o)

    # Crop png images to erase blank borders
    print '\nCrop png images to erase blank borders'
    crop_blank_edges(name1+'_resampled'+'_registered')
    crop_blank_edges(name2+'_resampled'+'_registered')
    crop_blank_edges(name3+'_resampled'+'_registered')

    # resize images to the same size as the WM atlas images
    print '\nResize images to the same size as the WM atlas images'
    resize_gm_png_to_wm_png_name(name1+'_resampled'+'_registered'+'_crop', name4)
    resize_gm_png_to_wm_png_name(name2+'_resampled'+'_registered'+'_crop', name4)
    resize_gm_png_to_wm_png_name(name3+'_resampled'+'_registered'+'_crop', name4)

    # tester correspondances entre les images
    print '\nTest dice coeffcient between the two images'
    status, output = sct.run('sct_dice_coefficient ' + name2 + '_resampled' + '_registered' + ext_o + ' ' + name4 + ext_o)
    print output

    # delete file imported at the beginning
    os.remove(fname1)
    os.remove(fname2)
    os.remove(fname3)
    os.remove(fname4)
    os.remove(slice)

    #anti_trans: input: greyscale_final_resampled_registered_crop_resized.png    output: greyscale_reg_no_trans.png (old: greyscale_final_reg_no_trans.png)
    print '\nReplacing transition pixel between zones...'
    anti_trans(fname10, list=[0,44,80,120,150,190,220,255], name_output=fname11)

    #copy left side on right side: input: greyscale_reg_no_trans.png  output: greyscale_reg_no_trans_sym.png
    print '\nCopying left side of the image on the right side with change of values...'
    antisym_im(fname11, name_output=fname12)


    #concatenation of GM and WM tracts: inputs: atlas_grays_cerv_sym_correc_r5.png  and  greyscale_reg_no_trans_sym.png  output: concatenation.png
    print '\nConcatenating WM and GM tracts...'
    concatenate_WM_and_GM(WM_file=fname13, GM_file=fname12, name_output='concatenation.png')

    #small hand corrections:  input: concatenation.png  output: concatenation_corrected.png

def concat_and_apply(inputs, dest, output_names, warps, interpolation='Linear'):
    # input = [input1, input2]
    # warps = [warp_1, warp_2]
    warp_str = ''
    for i in range(len(warps)):
        warp_str = ' '.join((warp_str, warps[-1-i]))
    cmd_0 = ('isct_ComposeMultiTransform 2 outwarp.nii.gz  ' + warp_str + ' -R ' + dest)
    sct.run(cmd_0)
    for j in range(len(inputs)):
        cmd_1 = ('isct_antsApplyTransforms -d 2 -i ' + inputs[j] + ' -o '+ output_names[j] + ' -n '+interpolation+' -t outwarp.nii.gz -r '+ dest)
        sct.run(cmd_1)

def save_nii_from_png(fname):
    path, file, ext = sct.extract_fname(fname)
    image = Image.open(fname).convert("L")
    arr = np.asarray(image)
    img = nibabel.Nifti1Image(arr, None)
    nibabel.save(img, file + ext_o)

#define header of output
def copy_hdr(input, dest):
    # Copy part of the header of dest into the header of intput
    cmd = ('fslcpgeom ' + dest +' '+ input + ' -d')
    sct.run(cmd)
    #change size of pixel by hand with set_q_form and set set_s_form

# creer fichier nifti de meme taille: interpoler pour mettre les 2 images a la bonne taille
def interpolate(fname, interpolation = 'Linear'):
    path, file, ext = sct.extract_fname(fname)
    cmd =('isct_c3d '+fname+' -interpolation '+interpolation + ' -resample 492x363x1 -o '+ file + '_resampled' + ext)
    sct.run(cmd)

def save_png_from_nii(fname):
    path, file, ext = sct.extract_fname(fname)
    data = nibabel.load(fname).get_data()
    sagittal = data[ :, :].T
    fig, ax = plt.subplots(1,1)
    ax.imshow(sagittal, cmap='gray', origin='lower')
    ax.set_title('sagittal')
    ax.set_axis_off()
    fig1 = plt.gcf()
    fig1.savefig(file + '.png', format='png')

# Crop blank edges coming from save_png_from_nii
# minus 102 along x and minus 75 along y (on each side)
def crop_blank_edges(name_png):
    img = Image.open(name_png+'.png')
    data = np.asarray(img)
    data_out = data[75:525, 104:717]
    im_out = Image.fromarray(data_out)
    im_out.save(name_png+'_crop'+'.png')

def resize_gm_png_to_wm_png_name(name_png, name_dest_png):
    img = Image.open(name_png+'.png')
    img_dest = Image.open(name_dest_png+'.png')
    img_resized = img.resize((img_dest.size[1],img_dest.size[0]))
    img_resized.save(name_png+'_resized.png')

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
#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()

