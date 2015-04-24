#!/usr/bin/env python
########################################################################################################################
#
#
# Compute TSNR using inputed anat.nii.gz and fmri.nii.gz files.
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
# Modified: 2015-03-12
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

from msct_parser import *
import sct_utils as sct


'''
##ORIGINAL SHELL SCRIPT

# motion correct the fmri data
echo Motion correct the fMRI data...
mcflirt -in fmri -out fmri_moco

# compute tsnr
echo Compute the tSNR...
fslmaths fmri_moco -Tmean fmri_moco_mean
fslmaths fmri_moco -Tstd fmri_moco_std
fslmaths fmri_moco_mean -div fmri_moco_std fmri_tsnr

# register tsnr to anatomic
echo Register tSNR to anatomic...
sct_c3d  anat.nii.gz fmri_tsnr.nii.gz -reslice-identity -o fmri_tsnr_reslice.nii.gz

# Remove temp files
echo Remove temporary files...
rm fmri_moco_std.nii.gz

echo Done!
'''

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
        sct.printv('Motion correct the fMRI data...', self.param.verbose, 'normal')
        path_fmri, fname_fmri, ext_fmri = sct.extract_fname(self.fmri)
        fname_fmri_moco = fname_fmri + '_moco'
        #TODO: replace sct.run() by sct.runProcess() if available in the current branch
        sct.run('mcflirt -in ' + fname_fmri + ' -out ' + fname_fmri_moco)

        # compute tsnr
        sct.printv('Compute the tSNR...', self.param.verbose, 'normal')
        fname_fmri_moco_mean = fname_fmri_moco + '_mean'
        sct.run('fslmaths ' + fname_fmri_moco + ' -Tmean ' + fname_fmri_moco_mean)
        fname_fmri_moco_std = fname_fmri_moco + '_std'
        sct.run('fslmaths ' + fname_fmri_moco + ' -Tstd ' + fname_fmri_moco_std)
        fname_fmri_tsnr = fname_fmri + '_tsnr'
        sct.run('fslmaths ' + fname_fmri_moco_mean + ' -div ' + fname_fmri_moco_std + ' ' + fname_fmri_tsnr)

        # register tsnr to anatomic
        sct.printv('Register tSNR to anatomic...', self.param.verbose, 'normal')
        sct.run('sct_c3d ' + self.anat + ' ' + fname_fmri_tsnr + '.nii.gz -reslice-identity -o ' + fname_fmri_tsnr + '_reslice.nii.gz')

        # Remove temp files
        sct.printv('Remove temporary files...', self.param.verbose, 'normal')
        sct.run('rm ' + fname_fmri_moco_std + '.nii.gz')

        sct.printv('Done!', self.param.verbose, 'info')


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
        parser.usage.set_description('Compute the tSNR given fMRI data and anatomic data files.')
        parser.add_option(name="-fmri",
                          type_value="file",
                          description="fMRI data file",
                          mandatory=True,
                          example='fmri.nii.gz')
        parser.add_option(name="-anat",
                          type_value="file",
                          description="anatomic data file",
                          mandatory=True,
                          example='anat.nii.gz')
        parser.add_option(name="-v",
                          type_value="int",
                          description="verbose",
                          mandatory=False,
                          example='1')

        arguments = parser.parse(sys.argv[1:])
        input_fmri = arguments["-fmri"]
        input_anat = arguments["-anat"]

        if "-v" in arguments:
            param.verbose = arguments["-v"]

        tsnr = Tsnr(param=param, fmri=input_fmri, anat=input_anat)
        tsnr.compute()





