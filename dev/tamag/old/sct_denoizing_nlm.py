#!/usr/bin/env python

import sys, commands
import numpy as np
from time import time
import nibabel as nib
from dipy.denoise.nlmeans import nlmeans
import os

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

from msct_parser import Parser
import sct_utils as sct

import matplotlib.pyplot as plt



# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.remove_temp_files = 1
        self.parameter = "Rician"
        self.file_to_denoise = ''
        self.output_file_name = ''



def main(file_to_denoise, param, output_file_name) :

    path, file, ext = sct.extract_fname(file_to_denoise)

    img = nib.load(file_to_denoise)
    hdr_0 = img.get_header()

    data = img.get_data()
    aff = img.get_affine()

    mask = data[:, :, :] > 80

    data = data[:, :, :]

    print("vol size", data.shape)

    t = time()

    sigma = np.std(data[~mask])

    if param.parameter == 'Rician':
        den = nlmeans(data, sigma=sigma, mask=mask, rician=True)
    else : den = nlmeans(data, sigma=sigma, mask=mask, rician=False)

    print("total time", time() - t)
    print("vol size", den.shape)


    axial_middle = data.shape[2] / 2

    before = data[:, :, axial_middle].T
    after = den[:, :, axial_middle].T

    diff_3d = np.absolute(den.astype('f8') - data.astype('f8'))
    difference = np.absolute(after.astype('f8') - before.astype('f8'))
    difference[~mask[:, :, axial_middle].T] = 0

    if param.verbose == 2 :
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(before, cmap='gray', origin='lower')
        ax[0].set_title('before')
        ax[1].imshow(after, cmap='gray', origin='lower')
        ax[1].set_title('after')
        ax[2].imshow(difference, cmap='gray', origin='lower')
        ax[2].set_title('difference')
        for i in range(3):
            ax[i].set_axis_off()

        plt.show()
        plt.savefig('denoised_S0.png', bbox_inches='tight')

    #Save files
    img_denoize = nib.Nifti1Image(den, None, hdr_0)
    img_diff = nib.Nifti1Image(diff_3d, None, hdr_0)
    if output_file_name != None :
        output_file_name =output_file_name
    else: output_file_name = file + '_denoised' + ext
    nib.save(img_denoize,output_file_name)
    nib.save(img_diff, file + '_difference' +ext)



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters


    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Utility function for labels.')
    parser.add_option(name="-i",
                      type_value='file',
                      description="List containing segmentation NIFTI file and label NIFTI files. They must be 3D. Names must be separated by commas without spaces.",
                      mandatory=True)
    parser.add_option(name="-p",
                      type_value="multiple_choice",
                      description="Type of supposed noise: Rician or Gaussian. Default is Rician.",
                      mandatory=False,
                      example=["Rician","Gaussian"],
                      default_value="Rician")

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Name of the output NIFTI image.",
                      mandatory=False)

    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files. Specify 0 to get access to temporary files.",
                      mandatory=False,
                      example=['0','1'],
                      default_value="1")
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="Verbose. 0: nothing. 1: basic. 2: extended.",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1', '2'])
    arguments = parser.parse(sys.argv[1:])

    parameter = arguments["-p"]
    remove_temp_files = int(arguments["-r"])
    verbose = int(arguments["-v"])

    if "-i" in arguments:
        file_to_denoise = arguments["-i"]
    else: file_to_denoize = None
    if "-o" in arguments:
        output_file_name = arguments["-o"]
    else: output_file_name = None

    param = Param()
    param.verbose = verbose
    param.remove_temp_files =remove_temp_files
    param.parameter = parameter

    main(file_to_denoise, param, output_file_name)
