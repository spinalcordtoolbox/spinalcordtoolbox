#!/usr/bin/env python
########################################################################################################################
#
#
# Utility functions useed for the segmentation of the gray matter
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# Modified: 2015-03-24
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

from math import sqrt

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
#from scipy.optimize import minimize

from msct_image import Image
import sct_utils as sct
from msct_parser import Parser


class Slice:
    def __init__(self, id=None, A=None, D=None, RtoM='', AM=None, DM=None, AM_flat=None, DM_flat=None, level=None):
        self.id = id
        self.A = A
        self.D = D
        self.RtoM = RtoM
        self.AM = AM
        self.DM = DM
        self.AM_flat = AM_flat
        self.DM_flat = DM_flat
        self.level = level

    def set(self, id=None, A=None, D=None, RtoM='', AM=None, DM=None, AM_flat=None, DM_flat=None, level=None):
        if id is not None:
            self.id = id
        if A is not None:
            self.A = A
        if D is not None:
            self.D = D
        if RtoM != '':
            self.RtoM = RtoM
        if AM is not None:
            self.AM = AM
        if DM is not None:
            self.DM = DM
        if AM_flat is not None:
            self.AM_flat = AM_flat
        if DM_flat is not None:
            self.DM_flat = DM_flat
        if level is not None:
            self.level = level

    def __repr__(self):
        s = '\nSlice #' + str(self.id)
        if self.level is not None:
            s += 'Level : ' + str(self.level)
        s += '\nAtlas : \n' + str(self.A) + '\nDecision : \n' + str(self.D) + '\nTransfor;ation to model space : ' + self.RtoM
        if self.AM is not None:
            s += '\nAtlas in the common model space: \n' + str(self.AM)
        if self.DM is not None:
             s += '\nDecision in the common model space: \n' + str(self.DM)
        return s


########################################################################################################################
######------------------------------------------------ FUNCTIONS -------------------------------------------------######
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
# Split a slice in two slices, used to deal with actual loss of data
def split(slice):
    left_slice = []
    right_slice = []
    column_length = slice.shape[1]
    i = 0
    for column in slice:
        if i < column_length / 2:
            left_slice.append(column)
        else:
            right_slice.insert(0, column)
        i += 1
    left_slice = np.asarray(left_slice)
    right_slice = np.asarray(right_slice)
    assert (left_slice.shape == right_slice.shape), \
        str(left_slice.shape) + '==' + str(right_slice.shape) + \
        'You should check that the first dim of your image (or slice) is an odd number'
    return left_slice, right_slice


# ----------------------------------------------------------------------------------------------------------------------
def show(coord_projected_img, pca, target, split=0):
    # Retrieving projected image from the mean image & its coordinates
    import copy

    img_reducted = copy.copy(pca.mean_image)
    for i in range(0, coord_projected_img.shape[0]):
        img_reducted += int(coord_projected_img[i][0]) * pca.W.T[i].reshape(pca.N, 1)

    if split :
        n = int(sqrt(pca.N * 2))
    else:
        n = int(sqrt(pca.N))
    if split:
        imgplot = plt.imshow(pca.mean_image.reshape(n, n / 2))
    else:
        imgplot = plt.imshow(pca.mean_image.reshape(n, n))
    imgplot.set_interpolation('nearest')
    imgplot.set_cmap('gray')
    plt.title('Mean Image')
    plt.show()
    if split:
        imgplot = plt.imshow(target.reshape(n, n / 2))
    else:
        imgplot = plt.imshow(target.reshape(n, n))
    imgplot.set_interpolation('nearest')
    #imgplot.set_cmap('gray')
    plt.title('Original Image')
    plt.show()
    if split:
        imgplot = plt.imshow(img_reducted.reshape(n, n / 2))
    else:
        imgplot = plt.imshow(img_reducted.reshape(n, n))
    imgplot.set_interpolation('nearest')
    #imgplot.set_cmap('gray')
    plt.title('Projected Image')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# save an image from an array, if the array correspond to a flatten image, the saved image will be square shaped
def save_image(im_array, im_name, path='', type='', verbose=1):
    if isinstance(im_array, list):
        n = int(sqrt(len(im_array)))
        im_data = np.asarray(im_array).reshape(n,n)
    else:
        im_data = np.asarray(im_array)
    im = Image(param=im_data,verbose=verbose)
    im.file_name = im_name
    im.ext = '.nii.gz'
    if path != '':
        im.path = path
    im.save(type=type)

def apply_ants_2D_rigid_transfo(fixed_im, moving_im, search_reg=True, transfo_type='Rigid', apply_transfo=True, transfo_name='', binary = True, path='./', inverse=0, verbose=0):
    import time
    try:
        transfo_dir = transfo_type.lower() + '_transformations'
        if transfo_dir not in os.listdir(path):
            sct.run('mkdir ' + path + transfo_dir)
        dir_name = 'tmp_reg_' +str(time.time())
        sct.run('mkdir ' + dir_name, verbose=verbose)
        os.chdir('./'+ dir_name)

        if binary:
            t = 'uint8'
        else:
            t = ''

        fixed_im_name = 'fixed_im'
        save_image(fixed_im, fixed_im_name, type=t, verbose=verbose)
        moving_im_name = 'moving_im'
        save_image(moving_im, moving_im_name, type=t, verbose=verbose)

        if search_reg:
            reg_interpolation = 'BSpline'
            gradientstep = 0.3  # 0.5
            metric = 'MeanSquares'
            metric_params = ',5'
            #metric = 'MI'
            #metric_params = ',1,2'
            niter = 20
            smooth = 0
            shrink = 1
            cmd_reg = 'sct_antsRegistration -d 2 -n ' + reg_interpolation + ' -t ' + transfo_type + '[' + str(gradientstep) + '] ' \
                      '-m ' + metric + '[' + fixed_im_name +'.nii.gz,' + moving_im_name + '.nii.gz ' + metric_params  + '] -o reg  -c ' + str(niter) + \
                      ' -s ' + str(smooth) + ' -f ' + str(shrink) + ' -v ' + str(verbose)

            sct.runProcess(cmd_reg, verbose=verbose)

            sct.run('cp reg0GenericAffine.mat ../' + path + transfo_dir + '/'+transfo_name, verbose=verbose)


        if apply_transfo:
            if not search_reg:
                sct.run('cp ../' + path + transfo_dir + '/' + transfo_name + ' ./reg0GenericAffine.mat ', verbose=verbose)

            if binary:
                applyTransfo_interpolation = 'NearestNeighbor'
            else:
                applyTransfo_interpolation = 'BSpline'

            cmd_apply = 'sct_antsApplyTransforms -d 2 -i ' + moving_im_name +'.nii.gz -o ' + moving_im_name + '_moved.nii.gz ' \
                        '-n ' + applyTransfo_interpolation + ' -t [reg0GenericAffine.mat,'+ str(inverse) +']  -r ' + fixed_im_name + '.nii.gz -v ' + str(verbose)

            status, output = sct.runProcess(cmd_apply, verbose=verbose)

            res_im = Image(moving_im_name + '_moved.nii.gz')
    except Exception, e:
        sct.printv('WARNING: AN ERROR OCCURRED WHEN DOING RIGID REGISTRATION USING ANTs',1 ,'warning')
        print e
    else:
        sct.printv('Removing temporary files ...',verbose = verbose, type='normal')
        os.chdir('..')
        sct.run('rm -rf ' + dir_name + '/', verbose=verbose)

    if apply_transfo:
        return res_im.data

# ----------------------------------------------------------------------------------------------------------------------
# Kronecker delta function
def kronecker_delta(x, y):
    if x == y:
        return 1
    else:
        return 0

# ------------------------------------------------------------------------------------------------------------------
# label-based cost function
def l0_norm(X, Y):
    return np.linalg.norm(X.flatten() - Y.flatten(), 0)

########################################################################################################################
######---------------------------------------------- PRETREATMENTS -----------------------------------------------######
########################################################################################################################
def crop_T2_star(dir):
    for subject_dir in os.listdir(dir):
        if os.path.isdir(subject_dir):
            t2star = ''
            sc_seg = ''
            seg_in = ''
            manual_seg = ''
            mask_box = ''
            seg_in_croped = ''
            manual_seg_croped = ''

            #print subject_dir

            '''
            #VERSION 1 OF THE PRE TREATMENTS

            mask_centerline = ''
            centerline = ''
            mask_box = ''
            croped = ''
            print subject_dir
            for file in os.listdir(dir + '/' + subject_dir):
                if 't2star.nii' in file and 'mask' not in file:
                    t2star = file
                elif '-mask_centerline.nii' in file:
                    mask_centerline = file
                elif 'mask_' in file:
                    mask_box = file
                elif '_centerline.nii' in file:
                    centerline = file
                elif '_croped' in file :
                    croped = file
            if t2star != '' and mask_centerline != '':
                path = dir + '/' + subject_dir + '/'
                print 'path : ', path
                os.chdir(path)
                t2star_path,t2star_name,ext = sct.extract_fname(t2star)
                if centerline == '':
                    sct.run('sct_get_centerline -i ' + t2star + ' -p '  + mask_centerline)
                if mask_box == '':
                    sct.run('sct_create_mask -i '  + t2star + ' -m centerline,'  + centerline +' -s 40 -f box' )
                if croped == '':
                    sct.run('sct_crop_image -i '  + t2star + ' -o '  + t2star_name + '_croped' + ext+ ' -m mask_' + t2star)
                os.chdir('..')
            '''



            '''
            #VERSION 2 OF THE PRE TREATMENTS



            for file in os.listdir(dir + '/' + subject_dir):
                if 't2star.nii' in file and 'mask' not in file and 'seg' not in file and 'IRP' not in file:
                    t2star = file
                    t2star_path,t2star_name,ext = sct.extract_fname(t2star)
                elif 'square_mask' in file and 'IRP' not in file:
                    mask_box = file
                elif '_seg' in file and 'in' not in file and 'croped' not in file and 'IRP' not in file:
                    sc_seg = file
                elif '_seg_in' in file and 'croped' not in file and 'IRP' not in file:
                    seg_in = file
                    seg_in_name = sct.extract_fname(seg_in)[1]
                elif '_croped' in file and 'IRP' not in file:
                    seg_in_croped = file
            if t2star != '' and sc_seg != '':
                path = dir + '/' + subject_dir + '/'
                print 'path : ', path
                os.chdir(path)

                try:

                    if seg_in == '':
                        sct.run('sct_crop_over_mask.py -i ' + t2star + ' -mask ' + sc_seg + ' -square 0 -o ' + t2star_name + '_seg_in')
                        seg_in = t2star_name + '_seg_in.nii.gz'
                        seg_in_name = t2star_name + '_seg_in'
                    if mask_box == '':
                        sct.run('sct_create_mask -i ' + t2star + ' -m center -s 70 -o ' + t2star_name + '_square_mask.nii.gz -f box' )
                        mask_box = t2star_name + '_square_mask.nii.gz'
                    if seg_in_croped == '':
                        sct.run('sct_crop_over_mask.py -i ' + seg_in + ' -mask ' + mask_box + ' -square 1 -o ' + seg_in_name + '_croped')
                    #os.chdir('..')

                except Exception,e:
                    sct.printv('WARNING: an error occured ... \n ' + str(e) ,1, 'warning')
                else:
                    print 'Done !'
                    #sct.run('rm -rf ./tmp_' + now)
                os.chdir('..')
            '''


            #VERSION 3 OF THE PRE TREATMENTS
            for file in os.listdir(dir + '/' + subject_dir):
                if 't2star.nii' in file and 'mask' not in file and 'seg' not in file and 'IRP' not in file:
                    t2star = file
                    t2star_path,t2star_name,ext = sct.extract_fname(t2star)
                elif 'square' in file and 'mask' in file and 'IRP' not in file:
                    mask_box = file
                elif '_seg' in file and 'in' not in file and 'croped' not in file and 'gm' not in file and 'IRP' not in file:
                    sc_seg = file
                elif '_seg_in' in file and 'croped' not in file and 'IRP' not in file:
                    seg_in = file
                    seg_in_name = sct.extract_fname(seg_in)[1]
                elif 'gm' in file and 'croped.nii' not in file and 'IRP' not in file:
                    manual_seg = file
                    print manual_seg
                    manual_seg_name = sct.extract_fname(manual_seg)[1]
                elif '_croped.nii' in file and 'IRP' not in file and 'gm' not in file:
                    seg_in_croped = file
                elif '_croped.nii' in file and 'gm' in file and 'IRP' not in file:
                    manual_seg_croped = file
                    print manual_seg_croped
            if t2star != '' and sc_seg != '':
                path = dir + '/' + subject_dir + '/'
                print 'path : ', path
                os.chdir(path)
                '''
                now = str(time.time())
                sct.run('mkdir tmp_'+ now)
                sct.run('cp ./' + t2star + ' ./tmp_'+now+'/'+t2star)
                sct.run('cp ./' + sc_seg + ' ./tmp_'+now+'/'+sc_seg)
                os.chdir('./tmp_'+now)
                '''
                try:

                    if seg_in == '':
                        sct.run('sct_crop_over_mask.py -i ' + t2star + ' -mask ' + sc_seg + ' -square 0 -o ' + t2star_name + '_seg_in')
                        seg_in = t2star_name + '_seg_in.nii.gz'
                        seg_in_name = t2star_name + '_seg_in'
                    if mask_box == '':
                        #sct.run('sct_create_mask -i ' + t2star + ' -m center -s 70 -o ' + t2star_name + '_square_mask.nii.gz -f box' )
                        #sct_create_mask -i errsm_05_t2star_seg_in.nii.gz -m centerline,errsm_05_t2star_seg_corrected.nii.gz -s 45 -f box -o errsm_05_t2star_mask_from_sc_seg.nii.gz
                        sct.run('sct_create_mask -i ' + seg_in + ' -m centerline,'+ sc_seg +' -s 43 -o ' + t2star_name + '_square_mask_from_sc_seg.nii.gz -f box' )
                        mask_box = t2star_name + '_square_mask_from_sc_seg.nii.gz'
                    if seg_in_croped == '':
                        sct.run('sct_crop_over_mask.py -i ' + seg_in + ' -mask ' + mask_box + ' -square 1 -o ' + seg_in_name + '_croped')
                    if manual_seg_croped == '':
                        sct.run('sct_crop_over_mask.py -i ' + manual_seg + ' -mask ' + mask_box + ' -square 1 -o ' + manual_seg_name + '_croped')

                    #os.chdir('..')

                except Exception,e:
                    sct.printv('WARNING: an error occured ... \n ' + str(e) ,1, 'warning')
                else:
                    print 'Done !'
                    #sct.run('rm -rf ./tmp_' + now)
                os.chdir('..')




if __name__ == "__main__":
        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Project all the input image slices on a PCA generated from set of t2star images')
        parser.add_option(name="-crop",
                          type_value="folder",
                          description="Path to the folder containing all your subjects' data",
                          mandatory=False,
                          example='dictionary/')

        arguments = parser.parse(sys.argv[1:])

        if "-crop" in arguments:
            crop_T2_star(arguments['-crop'])

