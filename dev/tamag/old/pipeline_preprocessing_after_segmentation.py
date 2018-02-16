#!/usr/bin/env python


import sys, os


# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
sys.path.append(path_sct + '/dev/tamag')

import sct_utils as sct
from glob import glob
from shutil import copy
from msct_parser import Parser

path = '/Users/tamag/data/data_for_template/marseille/temp'

# # DEFAULT PARAMETERS
# class Param:
#    ## The constructor
#    def __init__(self):
#        self.debug = 0
#        self.verbose = 1  # verbose
#        self.remove_temp_files = 1
#
#
# def main():

os.chdir(path)
list_dir = os.listdir(path)

for i in range(0, len(list_dir)):
    if os.path.isdir(list_dir[i]):
        list_dir_2 = os.listdir(path + '/' + list_dir[i])
        for j in range(len(list_dir_2)):
            if list_dir_2[j] == 'T2':
                # Going into last tmp folder
                # list_dir_3 = os.listdir(path + '/' + list_dir[i] + '/' + list_dir_2[j])
                # list_tmp_folder = [file for file in list_dir_3 if file.startswith('tmp')]
                # last_tmp_folder_name = list_tmp_folder[-1]
                # os.chdir(list_dir[i]+ '/' + list_dir_2[j]+'/'+last_tmp_folder_name)

                # Going into template creation:
                os.chdir(list_dir[i]+ '/' + list_dir_2[j]+'/template_creation')

                # Add label files and preprocess data for template registration
                print '\nPreprocessing data from: '+ list_dir[i]+ '/' + list_dir_2[j] + ' ...'
                name_seg_mod = 't2_crop_seg_mod_crop.nii.gz'
                sct.printv('sct_function_preprocessing.py -i *_t2_crop.nii.gz -l ' + name_seg_mod + ',up.nii.gz,down.nii.gz')
                os.system('sct_function_preprocessing.py -i *_t2_crop.nii.gz -l ' + name_seg_mod + ',up.nii.gz,down.nii.gz')
                name_output_straight = glob('*t2_crop_straight.nii.gz')[0]
                name_output_straight_normalized = glob('*t2_crop_straight_normalized.nii.gz')[0]

                # Copy resulting files into Results folder
                print '\nCopy output files into:/Users/tamag/data/data_for_template/Results_preprocess/T2'
                copy(name_output_straight, '/Users/tamag/data/data_for_template/Results_preprocess/T2/t2_' + list_dir[i]+'_crop_straight.nii.gz')
                copy(name_output_straight_normalized, '/Users/tamag/data/data_for_template/Results_preprocess/T2/t2_' + list_dir[i]+'_crop_straight_normalized.nii.gz')

                # Copy centerline and warping files to T1 folder
                copy('generated_centerline.nii.gz', '../../T1')
                copy('warp_curve2straight.nii.gz', '../../T1')
                copy('warp_straight2curve.nii.gz', '../../T1')

                # Display straightening
                sct.printv('fslview '+ name_output_straight + ' ' + name_output_straight_normalized + ' &')
                #os.system('fslview '+ name_output_straight + ' ' + name_output_straight_normalized + ' &')

                # Remove temporary file
                print('\nRemove temporary files...')
                #sct.run('rm -rf '+path_tmp)

                os.chdir('../../..')
#
#
# #=======================================================================================================================
# # Start program
# #=======================================================================================================================
# if __name__ == "__main__":
#     # initialize parameters
#
#
#     # Initialize the parser
#     parser = Parser(__file__)
#     parser.usage.set_description('Preprocessing of data: denoise, extract and fit the centerline, straighten the image using the fitted centerline and finally normalize the intensity.')
#     parser.add_option(name="-i",
#                      type_value='folder',
#                      description="Path of data to treat.",
#                      mandatory=True)
#
#     arguments = parser.parse(sys.argv[1:])
#
#     path = str(arguments('-i'))
#
#     param = Param()
#
#
#     main()
