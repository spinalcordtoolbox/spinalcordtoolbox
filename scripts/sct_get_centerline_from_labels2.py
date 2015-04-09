#!/usr/bin/env python


import numpy as np
import commands, sys


# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
sys.path.append('/home/tamag/code')

from msct_image import Image
from msct_parser import Parser
import nibabel
import os
import time
import sct_utils as sct
from sct_orientation import get_orientation, set_orientation
from sct_process_segmentation import b_spline_centerline
from scipy import interpolate
from msct_get_centerline_from_labels import ExtractCenterline
from sct_process_segmentation import extract_centerline
from sct_straighten_spinalcord import smooth_centerline
from copy import copy

# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.remove_temp_files = 1
        self.type_window = 'hanning'  # for smooth_centerline @sct_straighten_spinalcord
        self.window_length = 80  # for smooth_centerline @sct_straighten_spinalcord


def main(list_file, param, output_file_name=None, parameter = "binary_centerline", remove_temp_files = 1, verbose = 0):

#Process for a binary file as output:
    if parameter == "binary_centerline":
        path, file, ext = sct.extract_fname(list_file[0])

        # create temporary folder
        path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
        sct.run('mkdir '+path_tmp)

        # copy files into tmp folder
        for i in range(len(list_file)):
            file_temp = os.path.abspath(list_file[i])
            sct.run('cp '+file_temp+' '+path_tmp)

        # go to tmp folder
        os.chdir(path_tmp)

        # Concatenation of the files

        file_0 = nibabel.load(list_file[0])
        data_concatenation = file_0.get_data()
        hdr_0 = file_0.get_header()
        data_output = copy(data_concatenation)
        if len(list_file)>0:
            for i in range(1, len(list_file)):
                file_temp = nibabel.load(list_file[i])
                data_temp = file_temp.get_data()
                data_concatenation = data_concatenation + data_temp

        # Save concatenation as a file
        print '\nWrite NIFTI volumes...'
        img = nibabel.Nifti1Image(data_concatenation, None, hdr_0)
        nibabel.save(img,'concatenation_file.nii.gz')


        # Applying nurbs to the concatenation

        fname_output = extract_centerline('concatenation_file.nii.gz', remove_temp_files)

        # Display results


        # Save file as binary file



#Process for a text file as output:
    if parameter == "text_file" :
        print "\nText file process"

        # Concatenation of the files

        # Applying nurbs to the concatenation


        # Display results


        # Save file as txt file






#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters


    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Utility function for labels.')
    parser.add_option(name="-i",
                      type_value=[[','],'file'],
                      description="Segmentation NIFTI file and label NIFTI files. They must be 3D. Names must be separated by commas.",
                      mandatory=True)
    parser.add_option(name="-p",
                      type_value="multiple_choice",
                      description="Type of output wanted: \nbinary_centerline: return binary NIFTI file of the centerline \ntext_file: return text file with float coordinates according to z",
                      mandatory=False,
                      example=["binary_centerline","text_file"],
                      default_value="binary_centerline")

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Name of the output NIFTI image with the centerline, or of the output text file with the coordinates according to z.",
                      mandatory=False,
                      default_value="labels.nii.gz")

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

    #parameter = "binary_centerline"
    #remove_temp_files = 1
    # verbose = 0
    param = Param()
    parameter = arguments["-p"]
    remove_temp_files = int(arguments["-r"])
    verbose = int(arguments["-v"])

    if "-i" in arguments:
        list_file = arguments["-i"]
    else: list_file = None
    if "-o" in arguments:
        output_file_name = arguments["-o"]
    else: output_file_name = None

    #sct_obtain_centerline(segmentation_file="data_RPI_seg.nii.gz", label_file="labels_brainstem_completed.nii.gz", output_file_name="centerline_from_label_and_seg.nii.gz", parameter = "binary_centerline", remove_temp_files = 0, verbose = 0 )
    #sct_obtain_centerline(segmentation_file = "data_RPI_seg.nii.gz", label_file="labels_brainstem_completed.nii.gz", output_file_name="test_label_and_seg.txt", parameter = "text_file", remove_temp_files = 1, verbose = 0 )
    main(list_file, output_file_name, parameter, remove_temp_files, verbose, param)

#-i data_RPI_seg.nii.gz,labels_brainstem.nii.gz -o centerline_total.nii.gz -r 0 -v 1
#'nibabel.nifti1.Nifti1Image'