#!/usr/bin/env python
#
# This program returns the gray matter segmentation given anatomical, spinal cord segmentation and t2star images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Sara Dupont
# Modified: 2015-05-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import sct_utils as sct
import os
import time
import sys
import getopt
from msct_parser import *
from msct_image import Image
from sct_asman import Model, Param, GMsegSupervisedMethod
from msct_gmseg_utils import *


class Pretreatments:
    def __init__(self, target_fname, sc_seg_fname, t2_data=None):

        self.t2star = 't2star.nii.gz'
        self.sc_seg = 't2star_sc_seg.nii.gz'
        self.t2 = 't2.nii.gz'
        self.t2_seg = 't2_seg.nii.gz'
        self.t2_landmarks = 't2_landmarks.nii.gz'

        sct.run('cp ../' + target_fname + ' ./' + self.t2star)
        sct.run('cp ../' + sc_seg_fname + ' ./' + self.sc_seg)

        status, t2_star_orientation = sct.run('sct_orientation -i ' + self.t2star)
        self.original_orientation = t2_star_orientation[4:7]

        self.square_mask = crop_t2_star(self.t2star, self.sc_seg)

        self.treated_target = self.t2star[:-7] + '_seg_in_croped.nii.gz'

        self.level_fname = None
        if t2_data is not None:
            sct.run('cp ../' + t2_data[0] + ' ./' + self.t2)
            sct.run('cp ../' + t2_data[1] + ' ./' + self.t2_seg)
            sct.run('cp ../' + t2_data[2] + ' ./' + self.t2_landmarks)

            self.level_fname = compute_level_file(self.t2star, self.sc_seg, self.t2, self.t2_seg, self.t2_landmarks)


def main(target_fname, sc_seg_fname, t2_data, level_fname, param=None):
    # t2_im = Image(t2_fname) if t2_fname is not None else None
    sct.printv('\nBuilding the appearance model...', verbose=param.verbose, type='normal')
    model = Model(model_param=param, k=0.8)

    target_fname = check_file_to_niigz(target_fname)
    sc_seg_fname = check_file_to_niigz(sc_seg_fname)

    tmp_dir = 'tmp_' + sct.extract_fname(target_fname)[1] + '_' + time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir ' + tmp_dir)
    os.chdir(tmp_dir)

    level_to_use = None
    if level_fname is not None:
        t2_data = None
        if check_file_to_niigz('../' + level_fname):
            sct.run('cp ../' + level_fname + ' .')
            sct.run('sct_orientation -i ' + level_fname + ' -s IRP')
            level_to_use = sct.extract_fname(level_fname)[1] + '_IRP.nii.gz'
        else:
            level_to_use = level_fname

    sct.printv('\nDoing target pretreatments ...', verbose=param.verbose, type='normal')
    pretreat = Pretreatments(target_fname, sc_seg_fname, t2_data)
    if pretreat.level_fname is not None:
        level_to_use = pretreat.level_fname

    sct.printv('\nDoing target gray matter segmentation ...', verbose=param.verbose, type='normal')
    gm_seg = GMsegSupervisedMethod(pretreat.treated_target, level_to_use, model, gm_seg_param=param)

    sct.printv('\nDoing result post-treatments ...', verbose=param.verbose, type='normal')
    square_mask = Image(pretreat.square_mask)
    res_names = []
    for res_im in [gm_seg.res_wm_seg, gm_seg.res_gm_seg, gm_seg.corrected_wm_seg]:
        res_im_original_space = inverse_square_crop(res_im, square_mask)
        res_im_original_space.save()
        sct.run('sct_orientation -i ' + res_im_original_space.file_name + '.nii.gz -s RPI')
        res_name = sct.extract_fname(target_fname)[1] + res_im.file_name[len(pretreat.treated_target[:-7]):] + '.nii.gz'
        sct.run('cp ' + res_im_original_space.file_name + '_RPI.nii.gz ../' + res_name)
        res_names.append(res_name)

    os.chdir('..')

    sct.printv('Done! \nTo see the result, type :')
    sct.printv('fslview ' + target_fname + ' ' + res_names[0] + ' -l Red -t 0.4 ' + res_names[1] + ' -l Blue -t 0.4 &', param.verbose, 'info')



########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

if __name__ == "__main__":
    param = Param()
    input_target_fname = None
    input_sc_seg_fname = None
    input_t2_data = None
    input_level_fname = None
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_input = param.path_dictionary + "/errsm_34.nii.gz"
        fname_input = param.path_dictionary + "/errsm_34_seg_in.nii.gz"
    else:
        param_default = Param()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Project all the input image slices on a PCA generated from set of t2star images')
        parser.add_option(name="-i",
                          type_value="file",
                          description="T2star image you want to segment",
                          mandatory=True,
                          example='t2star.nii.gz')
        parser.add_option(name="-s",
                          type_value="file",
                          description="Spinal cord segmentation of the T2star target",
                          mandatory=True,
                          example='sc_seg.nii.gz')
        parser.add_option(name="-dic",
                          type_value="folder",
                          description="Path to the model data",
                          mandatory=True,
                          example='/home/jdoe/gm_seg_model_data/')
        parser.add_option(name="-t2",
                          type_value=[[','], 'file'],
                          description="T2 data associated to the input image : used to register the template on the T2star and get the vertebral levels"
                                      "In this order : t2 image, t2 segmentation, t2 landmarks (see: http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/)",
                          mandatory=False,
                          default_value=None,
                          example='t2.nii.gz,t2_seg.nii.gz,landmarks.nii.gz')
        parser.add_option(name="-l",
                          type_value="str",
                          description="Image containing level labels for the target or str indicating the level (if the target has only one slice)"
                                      "If -l is used, no need to provide t2 data",
                          mandatory=False,
                          example='MNI-Poly-AMU_level_IRP.nii.gz')
        parser.add_option(name="-v",
                          type_value="int",
                          description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                          mandatory=False,
                          default_value=0,
                          example='1')

        arguments = parser.parse(sys.argv[1:])
        input_target_fname = arguments["-i"]
        input_sc_seg_fname = arguments["-s"]
        param.path_dictionary = arguments["-dic"]
        param.todo_model = 'load'

        if "-t2" in arguments:
            input_t2_data = arguments["-t2"]
        if "-l" in arguments:
            input_level_fname = arguments["-l"]
        if "-v" in arguments:
            param.verbose = arguments["-v"]

    main(input_target_fname, input_sc_seg_fname, input_t2_data, input_level_fname, param)