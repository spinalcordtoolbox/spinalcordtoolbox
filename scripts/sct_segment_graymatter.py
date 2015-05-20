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
from sct_asman import Model, Param
from msct_gmseg_utils import *


class Pretreatments:
    def __init__(self, target_fname, sc_seg_fname, t2_fname=None):

        self.tmp_dir = 'tmp_' + sct.extract_fname(target_fname)[1] + '_' + time.strftime("%y%m%d%H%M%S")
        sct.run('mkdir ' + self.tmp_dir)
        os.chdir(self.tmp_dir)

        self.t2star = 't2star.nii.gz'
        self.sc_seg = 't2star_sc_seg.nii.gz'

        sct.run('cp ../' + target_fname + ' ./' + self.t2star)
        sct.run('cp ../' + sc_seg_fname + ' ./' + self.sc_seg)

        status, t2_star_orientation = sct.run('sct_orientation -i ' + self.t2star)
        self.original_orientation = t2_star_orientation[4:7]

        self.square_mask_IRP = crop_t2_star(self.t2star, self.sc_seg)

        os.chdir('..')



def main(target_fname, sc_seg_fname, t2_fname):
    # t2_im = Image(t2_fname) if t2_fname is not None else None
    pretreated = Pretreatments(target_fname, sc_seg_fname, t2_fname)



    croped_seg_in_im = Image(pretreated.tmp_dir + '/t2star_seg_in_croped.nii.gz')
    sq_mask = Image(pretreated.tmp_dir + '/' + pretreated.square_mask_IRP)
    test_im = inverse_square_crop(croped_seg_in_im, sq_mask)
    test_im.save()

    sct.run('sct_orientation -i ' + pretreated.tmp_dir + '/test.nii.gz -s RPI')



########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

if __name__ == "__main__":
    param = Param()
    input_target_fname = None
    input_sc_seg_fname = None
    input_t2_fname = None
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
                          mandatory=False,  # TODO: put back True when ready to use model
                          example='/home/jdoe/gm_seg_model_data/')
        parser.add_option(name="-t2",
                          type_value="file",
                          description="T2 image associated to the input image : used to register the template on the T2star and get the vertebral levels",
                          mandatory=False,
                          default_value=None,
                          example='t2.nii.gz')
        parser.add_option(name="-v",
                          type_value="int",
                          description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                          mandatory=False,
                          default_value=0,
                          example='1')

        arguments = parser.parse(sys.argv[1:])
        input_target_fname = arguments["-i"]
        input_sc_seg_fname = arguments["-s"]
        # param.path_dictionary = arguments["-dic"]
        param.todo_model = 'load'

        if "-t2" in arguments:
            input_t2_fname = arguments["-t2"]
        if "-v" in arguments:
            param.verbose = arguments["-v"]

    main(input_target_fname, input_sc_seg_fname, input_t2_fname)