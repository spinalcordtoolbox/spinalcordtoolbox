#!/usr/bin/env python


import sys


# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

from msct_parser import Parser
import sct_utils as sct

# DEFAULT PARAMETERS
class Param:
   ## The constructor
   def __init__(self):
       self.debug = 0
       self.verbose = 1  # verbose
       self.remove_temp_files = 1
       # self.type_window = 'hanning'  # for smooth_centerline @sct_straighten_spinalcord
       # self.window_length = 80  # for smooth_centerline @sct_straighten_spinalcord
       # self.algo_fitting = 'nurbs'
       # self.list_files = []
       # self.output_file_name = ''
       self.type_noise = 'Rician'

def main(input_anatomy_file, list_files, param, remove_temp_files = 1, verbose = 0) :

    path, file, ext = sct.extract_fname(input_anatomy_file)

    # Image denoising
    print '\nDenoising image ' + input_anatomy_file +'...'
    sct.run('sct_denoising_onlm.py -i '+ input_anatomy_file + ' -p ' + type_noise + ' -r ' + str(remove_temp_files) + ' -v ' + str(verbose))

    # Extract and fit centerline
    list_name_files = list_files[0]
    for i in range(1, len(list_files)):
        list_name_files = list_name_files + ',' + list_files[i]
    print '\nExtracting and fitting centerline...'
    sct.run('sct_get_centerline_from_labels -i '+ list_name_files + ' -r ' + str(remove_temp_files) + ' -v ' + str(verbose))

    # Straighten the image using the fitted centerline
    print '\nStraightening the image ' + input_anatomy_file + ' using the fitted centerline ' + 'generated_centerline.nii.gz'+ ' ...'
    sct.run('sct_straighten_spinalcord -i ' + input_anatomy_file + ' -c ' + 'generated_centerline.nii.gz' + ' -r ' + str(remove_temp_files) + ' -v ' + str(verbose))
    output_straighten_name = file + '_straight' +ext

    # Aplly transfo to the centerline
    print '\nApplying transformation to the centerline...'
    sct.run('sct_apply_transfo -i ' + 'generated_centerline.nii.gz' + ' -d ' + output_straighten_name + ' -w ' + 'warp_curve2straight.nii.gz' + ' -x ' + 'linear' + ' -v ' + str(verbose))

    # Normalize intensity of the image using the straightened centerline
    print '\nNormalizing intensity of the straightened image...'
    sct.run('sct_normalize.py -i ' + output_straighten_name + ' -c generated_centerline_reg.nii.gz' + ' -v ' + str(verbose))





#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters


    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Preprocessing of data: denoise, extract and fit the centerline, straighten the image using the fitted centerline and finally normalize the intensity.')
    parser.add_option(name="-i",
                     type_value='file',
                     description="Anatomic NIFTI image file.",
                     mandatory=True)
    parser.add_option(name="-l",
                     type_value=[[','],'file'],
                     description="List containing segmentation NIFTI file and label NIFTI files. They must be 3D. Names must be separated by commas without spaces. The list must at least contain a segmentation file.",
                     mandatory=True)
    # parser.add_option(name="-o",
    #                  type_value="file_output",
    #                  description="Name of the output NIFTI image with the centerline and of the output text file with the coordinates (z, x, y) (but text file will have '.txt' extension).",
    #                  mandatory=False,
    #                  default_value='generated_centerline.nii.gz')
    parser.add_option(name="-p",
                      type_value="multiple_choice",
                      description="Type of supposed noise: Rician or Gaussian. Default is Rician.",
                      mandatory=False,
                      example=["Rician","Gaussian"],
                      default_value="Rician")
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
    type_noise = arguments["-p"]

    if "-i" in arguments:
       input_anatomy_file = arguments["-i"]
    if "-l" in arguments:
       list_files = arguments["-l"]
    # if "-o" in arguments:
    #    output_file_name = arguments["-o"]
    # else: output_file_name = None

    param = Param()
    param.verbose = verbose
    param.remove_temp_files = remove_temp_files
    param.type_noise = type_noise

    main(input_anatomy_file, list_files, param, remove_temp_files, verbose)



