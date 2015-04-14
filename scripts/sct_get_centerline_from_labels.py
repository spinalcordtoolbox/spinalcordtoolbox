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
        self.parameter = "binary_centerline"
        self.list_file = []
        self.output_file_name = ''


def main(list_file, param, output_file_name=None, parameter = "binary_centerline", remove_temp_files = 1, verbose = 0):


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

    file_0 = load(list_file[0])
    data_concatenation = file_0.get_data()
    hdr_0 = file_0.get_header()
    if len(list_file)>0:
        for i in range(1, len(list_file)):
            file_temp = load(list_file[i])
            data_temp = file_temp.get_data()
            data_concatenation = data_concatenation + data_temp

    # Save concatenation as a file
    print '\nWrite NIFTI volumes...'
    img = Nifti1Image(data_concatenation, None, hdr_0)
    save(img,'concatenation_file.nii.gz')


    # Applying nurbs to the concatenation and save file as binary file
    fname_output = extract_centerline('concatenation_file.nii.gz', param, remove_temp_files=1)

    # Rename files after processing
    if output_file_name != None:
        output_file_name = output_file_name
    else : output_file_name = "generated_centerline.nii.gz"
    os.rename(fname_output, output_file_name)
    path_binary, file_binary, ext_binary = sct.extract_fname(output_file_name)
    os.rename('concatenation_file_centerline.txt', file_binary+'.txt')

# Process for a binary file as output:
    if parameter == "binary_centerline":
        #Copy binary file into parent folder
        sct.run('cp '+output_file_name+' ../')


# Process for a text file as output:
    if parameter == "text_file" :
        print "\nText file process"
        #path_binary, file_binary, ext_binary = sct.extract_fname(output_file_name)
        #Copy txt file into parent folder
        sct.run('cp '+file_binary+ '.txt'+ ' ../')

    # Remove temporary files
    if remove_temp_files:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)



    # Display results
    # The concatenate centerline and its fitted curve are displayed whithin extract_centerline







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
                      description="List containing segmentation NIFTI file and label NIFTI files. They must be 3D. Names must be separated by commas without spaces.",
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
        list_file = arguments["-i"]
    else: list_file = None
    if "-o" in arguments:
        output_file_name = arguments["-o"]
    else: output_file_name = None

    param = Param()
    param.verbose = verbose
    param.remove_temp_files =remove_temp_files

    main(list_file, param, output_file_name, parameter, remove_temp_files, verbose)

#-i data_RPI_seg.nii.gz,labels_brainstem.nii.gz -o centerline_total.nii.gz -r 0 -v 1