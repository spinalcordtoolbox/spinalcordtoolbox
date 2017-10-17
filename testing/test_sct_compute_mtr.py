#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_compute_mtr
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Stephanie Alley
#
#
# About the license: see the file LICENSE.TXT
#########################################################################################

def init(param_test):
     """
     Initialize class: param_test
     """
     # initialization
     default_args = ['-mt0 mt/mt0.nii.gz -mt1 mt/mt1.nii.gz']

    # assign default params
     if not param_test.args:
        param_test.args = default_args
    
     return param_test

def test_integrity(param_test):
     """
     Test integrity of function
     """
     param_test.output += '\nNot implemented.'
     
     return param_test
