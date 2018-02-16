#!/usr/bin/env python
#########################################################################################
# Convert dcm2nii using nin sequence from J. Finsterbusch
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
import os

import time


# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
import sct_utils as sct
from msct_parser import Parser

# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.path_data = ''



# MAIN
# ==========================================================================================
def main():

    # Initialization
    fsloutputdir = 'export FSLOUTPUTTYPE=NIFTI_GZ; '
    file_ordering = 'alternate'
    start_time = time.time()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Convert dcm2nii using nin sequence from J. Finsterbusch. '
                                 'Requires the software dcm2nii (from mricron).')
    parser.add_option(name="-i",
                      type_value="folder",
                      description="Path to dicom data.",
                      mandatory=True,
                      example="data/my_data")
    parser.add_option(name="-ord",
                      type_value="multiple_choice",
                      description="""File ordering: \nalternate: spine,brain,spine,brain... (with custom coil)\nbloc: spine,spine... brain,brain... (with head-neck coil)\n""",
                      mandatory=False,
                      default_value='bloc',
                      example=['alternate', 'bloc'])
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])

    arguments = parser.parse(sys.argv[1:])

    # get arguments
    path_data = arguments['-i']
    if '-ord' in arguments:
        file_ordering = arguments['-ord']
    remove_temp_files = int(arguments['-r'])
    verbose = int(arguments['-v'])

    path_tmp = sct.tmp_create(basename="nin_convert_dcm2nii", verbose=verbose)

    # go to temporary folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # list DICOM files
    file_data_list = os.listdir(path_data)

    # create another temp folder for conversion
    sct.run('mkdir tmp', verbose)

    # loop across files
    file_nii = []
    i = 0
    for i_file in file_data_list:
        # convert dicom to nifti and put in temporary folder
        status, output = sct.run('dcm2nii -o tmp/ -v n '+os.path.join(path_data, i_file), verbose)
        # change file name
        file_nii.append('data_'+str(i).zfill(4)+'.nii.gz')
        sct.run('mv tmp/*.nii.gz '+file_nii[i])
        # increment file index
        i = i+1

    # Merge data
    nb_files = len(file_data_list)
    if file_ordering == 'alternate':
        sct.run(fsloutputdir+'fslmerge -t data_spine '+' '.join([file_nii[i] for i in range(0, nb_files, 2)]))
        sct.run(fsloutputdir+'fslmerge -t data_brain '+' '.join([file_nii[i] for i in range(1, nb_files, 2)]))
    if file_ordering == 'bloc':
        sct.run(fsloutputdir+'fslmerge -t data_spine '+' '.join([file_nii[i] for i in range(0, nb_files/2)]))
        sct.run(fsloutputdir+'fslmerge -t data_brain '+' '.join([file_nii[i] for i in range(nb_files/2+1, nb_files)]))

    # come back
    os.chdir(curdir)

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    fname_data_spine = sct.generate_output_file(os.path.join(path_tmp, 'data_spine.nii.gz'), 'data_spine.nii.gz', verbose)
    fname_data_brain = sct.generate_output_file(os.path.join(path_tmp, 'data_brain.nii.gz'), 'data_brain.nii.gz', verbose)

    # Delete temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...', verbose)
        sct.run('rm -rf '+path_tmp, verbose)

    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s', verbose)
    sct.printv('\nTo view results, type:', verbose)
    sct.printv('fslview data_spine &', verbose, 'info')
    sct.printv('fslview data_brain &\n', verbose, 'info')


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
