#!/usr/bin/env python

# Detect Ponto-Medullary Junction
#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley
# Created: 2017-07-21
#
# About the license: see the file LICENSE.TXT

import os
import shutil
import sys
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import nibabel as nib

from msct_image import Image
from msct_parser import Parser
from sct_utils import tmp_create, extract_fname, slash_at_the_end, add_suffix, printv, run
from sct_image import get_orientation, set_orientation

def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Detection of the Ponto-Medullary Junction (PMJ).\n'
                                 ' This method is machine-learning based and adapted for T1w-like or T2w-like images.\n'
                                 ' If the PMJ is detected from the input image, a nifti mask with one voxel, with the value 50,\n'
                                 ' located at the predicted PMJ level, is output ("*_pmj.nii.gz").\n'
                                 ' If the PMJ is not detected, anything is output from this function.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast, if your contrast is not in the available options (t1, t2), use t1 (cord bright / CSF dark) or t2 (cord dark / CSF bright)",
                      mandatory=True,
                      example=["t1", "t2"])
    parser.add_option(name="-s",
                      type_value="file",
                      description="SC segmentation or SC centerline mask. To provide this mask could help the detection of the PMJ",
                      mandatory=False,
                      example="t2_seg.nii.gz")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      example="My_Output_Folder/")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files.",
                      mandatory=False,
                      default_value="1",
                      example=["0", "1"])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=["0", "1", "2"],
                      default_value="1")

    return parser

class DetectPMJ:
    def __init__(self, fname_im, contrast, fname_seg, path_out, verbose):

        self.fname_im = fname_im
        self.contrast = contrast

        self.fname_seg = fname_seg

        self.path_out = path_out

        self.verbose = verbose

        self.tmp_dir = tmp_create(verbose=self.verbose)  # path to tmp directory

        # to re-orient the data at the end if needed
        self.orientation_im = get_orientation(Image(self.fname_im))

        self.slice2D_im = extract_fname(self.fname_im)[1] + '_midSag.nii'
        self.slice2D_pmj = extract_fname(self.fname_im)[1] + '_midSag_pmj'

        path_script = os.path.dirname(__file__)
        path_sct = os.path.dirname(path_script)
        self.pmj_model = os.path.abspath(os.path.join(path_sct,
                                            'data/pmj_models',
                                            '{}_model'.format(self.contrast)))

    def apply(self):

        self.ifolder2tmp()

        self.orient2pir()

        self.extract_sagital_slice()

        self.detect()

#         # fill self.dct_metric --> for each key_metric: create an Image with zero values
#         self.init_metric_im()

#         # fill self.dct_im_seg --> extract axial slices from self.param.fname_im and self.param.fname_seg
#         self.extract_slices()

#         # compute texture
#         self.compute_texture()

#         # reorient data
#         if self.orientation_im != self.orientation_extraction:
#             self.reorient_data()

#         # mean across angles
#         self.mean_angle()

#         # save results to ofolder
#         self.tmp2ofolder()

#         return [self.param.path_results + self.fname_metric_lst[f] for f in self.fname_metric_lst]


    def detect(self):

        # os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"
        # cmd_pmj = 'isct_spine_detect -ctype=dpdt "%s" "%s" "%s"' % \
        #             (self.pmj_model, self.slice2D_im.split('.nii')[0], self.slice2D_pmj)
        cmd_pmj = 'isct_spine_detect '+self.pmj_model+' '+self.slice2D_im.split('.nii')[0]+' '+self.slice2D_pmj
        run(cmd_pmj, verbose=0)

        # convert .img and .hdr files to .nii
        img = nib.load(self.slice2D_pmj+'.hdr')
        nib.save(img, self.slice2D_pmj+'.nii')
        self.slice2D_pmj += '.nii'
        print self.slice2D_pmj

    def extract_sagital_slice(self): # function to extract a 2D sagital slice, used to do the detection

        if self.fname_seg is not None: # if the segmentation is provided, the 2D sagital slice is choosen accoding to the segmentation
            img_seg = Image(self.fname_seg)
            z_mid_slice = img_seg.data[:,int(img_seg.dim[1]/2),:]
            x_out = int(center_of_mass(z_mid_slice)[1])
            del img_seg
        else: # if the segmentation is not provided, the 2D sagital slice is choosen as the mid-sagital slice of the input image
            img = Image(self.fname_im)
            x_out = int(img.dim[2]/2)
            del img

        run('sct_crop_image -i '+self.fname_im+' -start '+str(x_out)+' -end '+str(x_out)+' -dim 2 -o '+self.slice2D_im)

    def orient2pir(self):
        
        if self.orientation_im != 'PIR': # open image and re-orient it to PIR if needed
            im_tmp = Image(self.fname_im)
            set_orientation(im_tmp, 'PIR', fname_out = ''.join(extract_fname(self.fname_im)[1:]))

            if self.fname_seg is not None:
                set_orientation(Image(self.fname_seg), 'PIR', fname_out = ''.join(extract_fname(self.fname_seg)[1:]))

    def ifolder2tmp(self):
        
        if self.fname_im is not None: # copy input image
            shutil.copy(self.fname_im, self.tmp_dir)
            self.fname_im = ''.join(extract_fname(self.fname_im)[1:])
        else:
            printv('ERROR: No input image', self.verbose, 'error')

        if self.fname_seg is not None: # copy segmentation image
            shutil.copy(self.fname_seg, self.tmp_dir)
            self.fname_seg = ''.join(extract_fname(self.fname_seg)[1:])
        else:
            printv('Warning: No segmentation image provided', self.verbose, 'warning')

        os.chdir(self.tmp_dir)  # go to tmp directory




#     def tmp2ofolder(self):

#         os.chdir('..')  # go back to original directory

#         printv('\nSave resulting files...', self.param.verbose, 'normal')
#         for f in self.fname_metric_lst:  # Copy from tmp folder to ofolder
#             shutil.copy(self.tmp_dir + self.fname_metric_lst[f],
#                         self.param.path_results + self.fname_metric_lst[f])



#     def mean_angle(self):

#         im_metric_lst = [self.fname_metric_lst[f].split('_' + str(self.param_glcm.distance) + '_')[0] + '_' for f in self.fname_metric_lst]
#         im_metric_lst = list(set(im_metric_lst))

#         printv('\nMean across angles...', self.param.verbose, 'normal')
#         extension = extract_fname(self.param.fname_im)[2]
#         for im_m in im_metric_lst:     # Loop across GLCM texture properties
#             # List images to mean
#             im2mean_lst = [im_m + str(self.param_glcm.distance) + '_' + a + extension for a in self.param_glcm.angle.split(',')]

#             # Average across angles and save it as wrk_folder/fnameIn_feature_distance_mean.extension
#             fname_out = im_m + str(self.param_glcm.distance) + '_mean' + extension
#             run('sct_image -i ' + ','.join(im2mean_lst) + ' -concat t -o ' + fname_out, error_exit='warning', raise_exception=True)
#             run('sct_maths -i ' + fname_out + ' -mean t -o ' + fname_out, error_exit='warning', raise_exception=True)
#             self.fname_metric_lst[im_m + str(self.param_glcm.distance) + '_mean'] = fname_out

#     def extract_slices(self):
#         # open image and re-orient it to RPI if needed
#         im, seg = Image(self.param.fname_im), Image(self.param.fname_seg)
#         if self.orientation_im != self.orientation_extraction:
#             im, seg = set_orientation(im, self.orientation_extraction), set_orientation(seg, self.orientation_extraction)

#         # extract axial slices in self.dct_im_seg
#         self.dct_im_seg['im'], self.dct_im_seg['seg'] = [im.data[:, :, z] for z in range(im.dim[2])], [seg.data[:, :, z] for z in range(im.dim[2])]

#     def init_metric_im(self):
#         # open image and re-orient it to RPI if needed
#         im_tmp = Image(self.param.fname_im)
#         if self.orientation_im != self.orientation_extraction:
#             im_tmp = set_orientation(im_tmp, self.orientation_extraction)

#         # create Image objects with zeros values for each output image needed
#         for m in self.metric_lst:
#             im_2save = im_tmp.copy()
#             im_2save.changeType(type='float64')
#             im_2save.data *= 0
#             fname_out = add_suffix(''.join(extract_fname(self.param.fname_im)[1:]), '_' + m)
#             im_2save.setFileName(fname_out)
#             im_2save.save()
#             self.fname_metric_lst[m] = fname_out

#     def compute_texture(self):

#         offset = int(self.param_glcm.distance)
#         printv('\nCompute texture metrics...', self.param.verbose, 'normal')

#         dct_metric = {}
#         for m in self.metric_lst:
#             dct_metric[m] = Image(self.fname_metric_lst[m])

#         timer = Timer(number_of_iteration=len(self.dct_im_seg['im']))
#         timer.start()

#         for im_z, seg_z, zz in zip(self.dct_im_seg['im'], self.dct_im_seg['seg'], range(len(self.dct_im_seg['im']))):
#             for xx in range(im_z.shape[0]):
#                 for yy in range(im_z.shape[1]):
#                     if not seg_z[xx, yy]:
#                         continue
#                     if xx < offset or yy < offset:
#                         continue
#                     if xx > (im_z.shape[0] - offset - 1) or yy > (im_z.shape[1] - offset - 1):
#                         continue  # to check if the whole glcm_window is in the axial_slice
#                     if False in np.unique(seg_z[xx - offset: xx + offset + 1, yy - offset: yy + offset + 1]):
#                         continue  # to check if the whole glcm_window is in the mask of the axial_slice

#                     glcm_window = im_z[xx - offset: xx + offset + 1, yy - offset: yy + offset + 1]
#                     glcm_window = glcm_window.astype(np.uint8)

#                     dct_glcm = {}
#                     for a in self.param_glcm.angle.split(','):  # compute the GLCM for self.param_glcm.distance and for each self.param_glcm.angle
#                         dct_glcm[a] = greycomatrix(glcm_window,
#                                                    [self.param_glcm.distance], [radians(int(a))],
#                                                    symmetric=self.param_glcm.symmetric,
#                                                    normed=self.param_glcm.normed)

#                     for m in self.metric_lst:  # compute the GLCM property (m.split('_')[0]) of the voxel xx,yy,zz
#                         dct_metric[m].data[xx, yy, zz] = greycoprops(dct_glcm[m.split('_')[2]], m.split('_')[0])[0][0]

#             timer.add_iteration()

#         timer.stop()

#         for m in self.metric_lst:
#             dct_metric[m].setFileName(self.fname_metric_lst[m])
#             dct_metric[m].save()

#     def reorient_data(self):
#         for f in self.fname_metric_lst:
#             im = Image(self.fname_metric_lst[f])
#             im = set_orientation(im, self.orientation_im)
#             im.setFileName(self.fname_metric_lst[f])
#             im.save()

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # get parser
    parser = get_parser()
    arguments = parser.parse(args)

    # set param arguments ad inputted by user
    fname_im = arguments["-i"]
    contrast = arguments["-c"]

    if '-s' in arguments:
        fname_seg = arguments['-s']
        if not os.path.isfile(fname_seg):
            fname_seg = None
            printv('WARNING: -s input file: "' + arguments['-s'] + '" does not exist.\nDetecting PMJ without using segmentation information', 1, 'warning')
    else:
        fname_seg = None

    if '-ofolder' in arguments:
        path_results = slash_at_the_end(arguments["-ofolder"], slash=1)
        if not os.path.isdir(path_results) and os.path.exists(path_results):
            printv("ERROR output directory %s is not a valid directory" % path_results, 1, 'error')
        if not os.path.exists(path_results):
            os.makedirs(path_results)
    else:
        path_results = './'

    if '-r' in arguments:
        rm_tmp = bool(int(arguments['-r']))
    else:
        rm_tmp = True

    if '-v' in arguments:
        verbose = int(arguments['-v'])
    else:
        verbose = '1'

    # Initialize DetectPMJ
    detector = DetectPMJ(fname_im=fname_im, contrast=contrast, fname_seg=fname_seg, path_out=path_results, verbose=verbose)
    # run the extraction
    fname_out = detector.apply()

    # # remove tmp_dir
    # if rm_tmp:
    #     shutil.rmtree(tmp_dir)

    # printv('\nDone! To view results, type:', verbose)
    # printv('fslview ' + arguments["-i"] + ' ' + fname_out + ' -l Red -t 0.7 & \n', verbose, 'info')

    # """
    #   - output a png with red dot : cf GM seg
    #   - remove abspath? self.pmj_model = os.path.abspath
    # """


if __name__ == "__main__":
    main()