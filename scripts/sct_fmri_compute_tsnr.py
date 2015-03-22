#!/usr/bin/env python
########################################################################################################################
#
#
# Compute TSNR using inputed anat.nii.gz and fmri.nii.gz files.
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
# Created: 2015-03-12
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

from msct_parser import *
import sct_utils as sct
import sys


class Param:
    def __init__(self):
        self.debug = 0
        self.verbose = 1

########################################################################################################################
######------------------------------------------------- Classes --------------------------------------------------######
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# TSNR -----------------------------------------------------------------------------------------------------------------
class Tsnr:
    def __init__(self, param=None, fmri=None, anat=None):
        if param is not None:
            self.param = param
        else:
            self.param = Param()
        self.fmri = fmri
        self.anat = anat

    def compute(self):

        # motion correct the fmri data
        sct.printv('\nMotion correct the fMRI data...', self.param.verbose, 'normal')
        path_fmri, fname_fmri, ext_fmri = sct.extract_fname(self.fmri)
        fname_fmri_moco = fname_fmri + '_moco'
        sct.runProcess('mcflirt -in ' + sct.slash_at_the_end(path_fmri) + fname_fmri + ' -out ' + fname_fmri_moco)

        # compute tsnr
        sct.printv('\nCompute the tSNR...', self.param.verbose, 'normal')
        fname_fmri_moco_mean = fname_fmri_moco + '_mean'
        sct.runProcess('fslmaths ' + fname_fmri_moco + ' -Tmean ' + fname_fmri_moco_mean)
        fname_fmri_moco_std = fname_fmri_moco + '_std'
        sct.runProcess('fslmaths ' + fname_fmri_moco + ' -Tstd ' + fname_fmri_moco_std)
        fname_fmri_tsnr = fname_fmri + '_tsnr'
        sct.runProcess('fslmaths ' + fname_fmri_moco_mean + ' -div ' + fname_fmri_moco_std + ' ' + fname_fmri_tsnr)

        # Remove temp files
        sct.printv('\nRemove temporary files...', self.param.verbose, 'normal')
        sct.runProcess('rm ' + fname_fmri_moco_std + '.nii.gz')

        # to view results
        sct.printv('\nDone! To view results, type:', self.param.verbose, 'normal')
        sct.printv('fslview '+fname_fmri_tsnr+' &\n', self.param.verbose, 'info')



########################################################################################################################
######-------------------------------------------------  MAIN   --------------------------------------------------######
########################################################################################################################

if __name__ == "__main__":
    param = Param()

    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
    else:
        param_default = Param()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Compute temporal SNR (tSNR) in fMRI time series.')
        parser.add_option(name="-i",
                          type_value="file",
                          description="fMRI data",
                          mandatory=True,
                          example='fmri.nii.gz')
        parser.add_option(name="-v",
                          type_value="int",
                          description="verbose",
                          mandatory=False)

        arguments = parser.parse(sys.argv[1:])
        input_fmri = arguments["-i"]

        if "-v" in arguments:
            param.verbose = arguments["-v"]

        tsnr = Tsnr(param=param, fmri=input_fmri)
        tsnr.compute()





