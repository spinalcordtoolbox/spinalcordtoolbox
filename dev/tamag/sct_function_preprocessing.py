#!/usr/bin/env python


import commands, sys


# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')

from msct_parser import Parser
from nibabel import load, save, Nifti1Image
import os
import time
import sct_utils as sct
from sct_process_segmentation import extract_centerline
from sct_orientation import get_orientation

# DEFAULT PARAMETERS
class Param:
   ## The constructor
   def __init__(self):
       self.debug = 0
       self.verbose = 1  # verbose
       self.remove_temp_files = 1
       self.type_window = 'hanning'  # for smooth_centerline @sct_straighten_spinalcord
       self.window_length = 80  # for smooth_centerline @sct_straighten_spinalcord
       self.algo_fitting = 'nurbs'
       # self.parameter = "binary_centerline"
       self.list_file = []
       self.output_file_name = ''
       self.type_noise = 'Rician'

def main() :
    debug = param.debug
    verbose = param.verbose
    remove_temp_files = param.remove_temp_files
    type_window = param.type_window
    window_length = param.window_length
    algo_fitting = param.algo_fitting
    list_file = param.list_file
    self.output_file_name
    type_noise = param.type_noise



    # Image denoising
    sct.run('sct_denoising_nlm.py -i '+ input_file + ' -p ' + type_noise + ' -r ' + remove_temp_files)

    # Extract and fit centerline
    sct.run('sct_denoising_nlm.py -i '+ input_file + ' -p ' + type_noise + ' -r ' + remove_temp_files)
    # sct_get_centerline_from_labels -i <list_file>

    # Straighten the image using the fitted centerline


    # Aplly transfo to the centerline


    # Normalize intensity of the image using the straightened centerline





































#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters


    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute a centerline from a list of segmentation and label files. It concatenates the parts, then extract the centerline. The output is a NIFTI image and a text file with the float coordinates (z, x, y) of the centerline.')
      parser.add_option(name="-i",
                     type_value='file',
                     description="Image NIFTI file.",
                     mandatory=True)
    parser.add_option(name="-l",
                     type_value=[[','],'file'],
                     description="List containing segmentation NIFTI file and label NIFTI files. They must be 3D. Names must be separated by commas without spaces.",
                     mandatory=True)
    parser.add_option(name="-o",
                     type_value="file_output",
                     description="Name of the output NIFTI image with the centerline and of the output text file with the coordinates (z, x, y). The orientation is RPI.",
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

    remove_temp_files = int(arguments["-r"])
    verbose = int(arguments["-v"])


    if "-i" in arguments:
        input_file = arguments["-i"]
    if "-l" in arguments:
       list_file = arguments["-i"]
    else: list_file = None
    if "-o" in arguments:
       output_file_name = arguments["-o"]
    else: output_file_name = None

    param = Param()
    param.verbose = verbose
    param.remove_temp_files =remove_temp_files

    main(input_file, list_file, param, output_file_name, remove_temp_files, verbose)