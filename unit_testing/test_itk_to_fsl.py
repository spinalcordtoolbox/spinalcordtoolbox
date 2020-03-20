#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for converting itk to absolute fsl warps


from __future__ import print_function, absolute_import

import os
import sys
import pytest
import numpy as np
import tempfile
import subprocess

from spinalcordtoolbox import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))

from spinalcordtoolbox.image import Image

import spinalcordtoolbox.testing.create_test_data as ctd
from sct_maths import mutual_information

def test_itk_to_fsl():
  tempfiles = [tempfile.NamedTemporaryFile(suffix = ".nii.gz") for i in range(5) ]

  seg_file, itk_warp_file, fsl_warp_file, itk_out_file, fsl_out_file = \
    [ tf.name for tf in tempfiles ]
  
  seg = ctd.dummy_segmentation([50,50,50])
  warp = ctd.dummy_deformation([50,50,50])

  seg.save(seg_file)
  warp.save(itk_warp_file)

  subprocess.run("sct_image -i {} -to-fsl {} -o {}"
                 "".format(itk_warp_file, seg_file, fsl_warp_file),
                 shell = True)

  subprocess.run("sct_apply_transfo -i {} -d {} -w {} -o {}"
                 "".format(seg_file, seg_file, itk_warp_file, itk_out_file),
                 shell = True)

  subprocess.run("applywarp -i {} -r {} -w {} -o {} --abs"
                 "".format(seg_file, seg_file, fsl_warp_file, fsl_out_file),
                 shell = True)

  itk_out = Image(itk_out_file)
  fsl_out = Image(fsl_out_file)

  itk_data_crop = itk_out.data[3:-3, 3:-3, 3:-3].ravel()
  fsl_data_crop = fsl_out.data[3:-3, 3:-3, 3:-3].ravel()

  joint_mask = np.logical_or(itk_data_crop > .01, fsl_data_crop > 0.01)

  mi = mutual_information(itk_data_crop[joint_mask], fsl_data_crop[joint_mask]
                          , normalized = True)

  assert mi > 0.5, "The itk to fsl transformations weren't similar enough"
  
