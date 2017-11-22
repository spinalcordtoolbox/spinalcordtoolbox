#!/usr/bin/env python


import commands, sys


# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

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


def main(list_file, param, output_file_name=None, remove_temp_files = 1, verbose = 0):

    path, file, ext = sct.extract_fname(list_file[0])

    path_tmp = sct.tmp_create(basename="centerline_from_labels", verbose=verbose)

    # copy files into tmp folder
    sct.printv('\nCopy files into tmp folder...', verbose)
    for i in range(len(list_file)):
       file_temp = os.path.abspath(list_file[i])
       sct.run('cp '+file_temp+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)

    ## Concatenation of the files

    # Concatenation : sum of matrices
    file_0 = load(file+ext)
    data_concatenation = file_0.get_data()
    hdr_0 = file_0.get_header()
    orientation_file_0 = get_orientation(list_file[0])
    if len(list_file)>0:
       for i in range(1, len(list_file)):
           orientation_file_temp = get_orientation(list_file[i])
           if orientation_file_0 != orientation_file_temp :
               print "ERROR: The files ", list_file[0], " and ", list_file[i], " are not in the same orientation. Use sct_orientation to change the orientation of a file."
               sys.exit(2)
           file_temp = load(list_file[i])
           data_temp = file_temp.get_data()
           data_concatenation = data_concatenation + data_temp

    # Save concatenation as a file
    print '\nWrite NIFTI volumes...'
    img = Nifti1Image(data_concatenation, None, hdr_0)
    save(img,'concatenation_file.nii.gz')


    # Applying nurbs to the concatenation and save file as binary file
    fname_output = extract_centerline('concatenation_file.nii.gz', remove_temp_files = remove_temp_files, verbose = verbose, algo_fitting=param.algo_fitting, type_window=param.type_window, window_length=param.window_length)

    # Rename files after processing
    if output_file_name != None:
       output_file_name = output_file_name
    else : output_file_name = "generated_centerline.nii.gz"

    os.rename(fname_output, output_file_name)
    path_binary, file_binary, ext_binary = sct.extract_fname(output_file_name)
    os.rename('concatenation_file_centerline.txt', file_binary+'.txt')

    # Process for a binary file as output:
    sct.run('cp '+output_file_name+' ../')

    # Process for a text file as output:
    sct.run('cp '+file_binary+ '.txt'+ ' ../')

    os.chdir('../')
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
   parser.usage.set_description('Compute a centerline from a list of segmentation and label files. It concatenates the parts, then extract the centerline. The output is a NIFTI image and a text file with the float coordinates (z, x, y) of the centerline.')
   parser.add_option(name="-i",
                     type_value=[[','],'file'],
                     description="List containing segmentation NIFTI file and label NIFTI files. They must be 3D. Names must be separated by commas without spaces.",
                     mandatory=True,
                     example= "data_seg.nii.gz,label1.nii.gz,label2.nii.gz")
   parser.add_option(name="-o",
                     type_value="file_output",
                     description="Name of the output NIFTI image with the centerline and of the output text file with the coordinates (z, x, y) (but text file will have '.txt' extension).",
                     mandatory=False,
                     default_value='generated_centerline.nii.gz')
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
       list_file = arguments["-i"]
   else: list_file = None
   if "-o" in arguments:
       output_file_name = arguments["-o"]
   else: output_file_name = None

   param = Param()
   param.verbose = verbose
   param.remove_temp_files =remove_temp_files

   main(list_file, param, output_file_name, remove_temp_files, verbose)
