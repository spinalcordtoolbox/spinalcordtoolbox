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
import commands

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

        self.pmj_model = os.path.join((commands.getoutput('$SCT_DIR')).split(': ')[1],
                                            'data/pmj_models',
                                            '{}_model'.format(self.contrast))

        self.threshold = -0.75 if self.contrast=='t1' else 0.8

        self.fname_out = extract_fname(self.fname_im)[1] + '_pmj.nii.gz'

    def apply(self):

        self.ifolder2tmp()

        self.orient2pir()

        self.extract_sagital_slice()

        self.detect()

        self.get_max_position()

        self.generate_mask_pmj()

        # save results to ofolder
        return self.tmp2ofolder()

    def tmp2ofolder(self):

        os.chdir('..')  # go back to original directory

        if self.pa_coord != -1:
            printv('\nSave resulting file...', self.verbose, 'normal')
            shutil.copy(os.path.abspath(os.path.join(self.tmp_dir, self.fname_out)),
                            os.path.abspath(os.path.join(self.path_out, self.fname_out)))

            return os.path.join(self.path_out, self.fname_out), self.tmp_dir
        
        else:
            return None, self.tmp_dir

    def generate_mask_pmj(self):

        if self.pa_coord != -1:
            im = Image(''.join(extract_fname(self.fname_im)[1:]))
            im_mask = im.copy()
            im_mask.data *= 0

            im_mask.data[self.pa_coord, self.is_coord, self.rl_coord] = 50

            im_mask.setFileName(self.fname_out)

            im_mask = set_orientation(im_mask, self.orientation_im, fname_out = self.fname_out)

            im_mask.save()

    def get_max_position(self):

        img_pred = Image(self.slice2D_pmj)
        
        if True in np.unique(img_pred.data > self.threshold):
            img_pred_maxValue = np.max(img_pred.data)
            self.pa_coord, self.is_coord = np.where(img_pred.data==img_pred_maxValue)[0][0], np.where(img_pred.data==img_pred_maxValue)[1][0]

        else:
            self.pa_coord, self.is_coord = -1, -1

        del img_pred

    def detect(self):

        os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"
        cmd_pmj = 'isct_spine_detect "%s" "%s" "%s"' % \
                    (self.pmj_model, self.slice2D_im.split('.nii')[0], self.slice2D_pmj)

        run(cmd_pmj, verbose=0)

        # convert .img and .hdr files to .nii
        img = nib.load(self.slice2D_pmj+'_svm.hdr')
        nib.save(img, self.slice2D_pmj+'.nii')
        self.slice2D_pmj += '.nii'

    def extract_sagital_slice(self): # function to extract a 2D sagital slice, used to do the detection

        if self.fname_seg is not None: # if the segmentation is provided, the 2D sagital slice is choosen accoding to the segmentation
            img_seg = Image(self.fname_seg)
            z_mid_slice = img_seg.data[:,int(img_seg.dim[1]/2),:]
            self.rl_coord = int(center_of_mass(z_mid_slice)[1])
            del img_seg
        else: # if the segmentation is not provided, the 2D sagital slice is choosen as the mid-sagital slice of the input image
            img = Image(self.fname_im)
            self.rl_coord = int(img.dim[2]/2)
            del img

        run('sct_crop_image -i '+self.fname_im+' -start '+str(self.rl_coord)+' -end '+str(self.rl_coord)+' -dim 2 -o '+self.slice2D_im)

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
    fname_out, tmp_dir = detector.apply()

    # remove tmp_dir
    if rm_tmp:
        shutil.rmtree(tmp_dir)

    if fname_out is not None:
        printv('\nDone! To view results, type:', verbose)
        printv('fslview ' + arguments["-i"] + ' ' + fname_out + ' -l Red -t 0.7 & \n', verbose, 'info')

    # """
    #   - output a png with red dot : cf GM seg
    # """


if __name__ == "__main__":
    main()