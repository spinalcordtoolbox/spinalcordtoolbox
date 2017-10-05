#!/usr/bin/env python

import sys, commands
import numpy as np
from time import time
import nibabel as nib
from msct_parser import Parser
import sct_utils as sct

# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')


# DEFAULT PARAMETERS
class Param:
    # The constructor
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

    if min(data.shape) <= 5:
        sct.printv('One of the image dimensions is <= 5 : reducing the size of the block radius.')
        block_radius = min(data.shape) - 1
    else:
        block_radius = 5  # default value

    # Process for manual detecting of background
    # mask = data[:, :, :] > noise_threshold
    # data = data[:, :, :]

    from dipy.denoise.nlmeans import nlmeans

    if '-std' in arguments:
        sigma = std_noise
        # Application of NLM filter to the image
        sct.printv('Applying Non-local mean filter...')
        if param.parameter == 'Rician':
            den = nlmeans(data, sigma=sigma, mask=None, rician=True, block_radius=block_radius)
        else :
            den = nlmeans(data, sigma=sigma, mask=None, rician=False, block_radius=block_radius)
    else:
        # # Process for manual detecting of background
        mask = data > noise_threshold
        sigma = np.std(data[~mask])
        # Application of NLM filter to the image
        sct.printv('Applying Non-local mean filter...')
        if param.parameter == 'Rician':
            den = nlmeans(data, sigma=sigma, mask=mask, rician=True, block_radius=block_radius)
        else:
            den = nlmeans(data, sigma=sigma, mask=mask, rician=False, block_radius=block_radius)

    t = time()
    sct.printv("total time", time() - t))
    sct.printv("vol size", den.shape))

    axial_middle = data.shape[2] / 2

    before = data[:, :, axial_middle].T
    after = den[:, :, axial_middle].T

    diff_3d = np.absolute(den.astype('f8') - data.astype('f8'))
    difference = np.absolute(after.astype('f8') - before.astype('f8'))
    if '-std' not in arguments:
        difference[~mask[:, :, axial_middle].T] = 0

    if param.verbose == 2 :
        import matplotlib.pyplot as plt
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

    # Save files
    img_denoise = nib.Nifti1Image(den, None, hdr_0)
    img_diff = nib.Nifti1Image(diff_3d, None, hdr_0)
    if output_file_name != None :
        output_file_name = output_file_name
    else:
        output_file_name = file + '_denoised' + ext
    nib.save(img_denoise, output_file_name)
    nib.save(img_diff, file + '_difference' + ext)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Utility function to denoise images. (Return the denoised image and also the difference between the input and the output.)')
    parser.add_option(name="-i",
                      type_value='file',
                      description="Input NIFTI image to be denoised.",
                      mandatory=True,
                      example='image_input.nii.gz')
    parser.add_option(name="-p",
                      type_value="multiple_choice",
                      description="Type of supposed noise: Rician or Gaussian. Default is Rician.",
                      mandatory=False,
                      example=["Rician", "Gaussian"],
                      default_value="Rician")
    parser.add_option(name="-d",
                      type_value="int",
                      description="Threshold value for what to be considered as noise. The standard deviation of the noise is calculated for values below this limit. Not relevant if -std value is precised.\n",
                      mandatory=False,
                      default_value="80")
    parser.add_option(name="-std",
                      type_value="float",
                      description="Standard deviation of the noise. If not precised, it is calculated using a background of point of values below the threshold value (parameter d).",
                      mandatory=False)
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Name of the output NIFTI image.",
                      mandatory=False)
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files. Specify 0 to get access to temporary files.",
                      mandatory=False,
                      example=['0', '1'],
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
    noise_threshold = int(arguments['-d'])

    if "-i" in arguments:
        file_to_denoise = arguments["-i"]
    else:
        file_to_denoise = None
    if "-o" in arguments:
        output_file_name = arguments["-o"]
    else:
        output_file_name = None
    if "-std" in arguments:
        std_noise = float(arguments['-std'])

    param = Param()
    param.verbose = verbose
    param.remove_temp_files = remove_temp_files
    param.parameter = parameter

    main(file_to_denoise, param, output_file_name)
