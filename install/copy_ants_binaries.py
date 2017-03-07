#!/usr/bin/env python
#
# Copy ANTs binaries into SCT folder.
#
# Author: Julien Cohen-Adad

# TODO: remove quick fix with folder_sct_temp
from __future__ import print_function

import os
import getopt
import sys
import shutil


def usage():
    print("""
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Copy ANTs binaries for packaging. Will output binaries in the local folder antsbin/

USAGE
  """+os.path.basename(__file__)+""" -f <folder_ants> -s <os_name>

MANDATORY ARGUMENTS
  -f <folder_ants>      antsbin folder (do not include slash at the end)

EXAMPLE
  """+os.path.basename(__file__)+""" -f ~/antsbin/bin\n""")

    sys.exit(2)

folder_bin = 'bin'
prefix_sct = 'isct_'
listOS = ['osx', 'linux']
list_file_ants = ['antsApplyTransforms', 'antsRegistration', 'antsSliceRegularizedRegistration', 'ComposeMultiTransform']
output_folder = 'antsbin/'


try:
    opts, args = getopt.getopt(sys.argv[1:], 'hf:')
except getopt.GetoptError as err:
    print(str(err))
    usage()
if not opts:
    usage()
for opt, arg in opts:
    if opt == '-h':
        usage()
    elif opt in ('-f'):
        folder_ants = str(arg)+'/bin/'

# check if ANTs folder exists
if not os.path.exists(folder_ants):
    print('\nERROR: Path '+folder_ants+' does not exist.\n')

# build destination path name
os.makedirs(output_folder)

# loop across ANTs files
for file_ants in list_file_ants:
    # check if file exists
    if not os.path.exists(folder_ants+file_ants):
        print('\nERROR: File '+folder_ants+file_ants+' does not exist.\n')
    # copy data
    print('Copying: '+file_ants)
    shutil.copyfile(folder_ants+file_ants, output_folder+prefix_sct+file_ants)
#    os.cosct.run('cp '+folder_ants+file_ants+' '+path_bin+prefix_sct+file_ants, 1)

print("Done!\n")
