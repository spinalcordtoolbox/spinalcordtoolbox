#!/usr/bin/env python
#
# Copy ANTs binaries into SCT folder. 
#
# Author: Julien Cohen-Adad
# modified: 2014-11-01

# TODO: remove quick fix with folder_sct_temp

import sys, io, os, getopt, shutil

path_sct = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(path_sct, 'scripts'))
import sct_utils as sct


# usage
#=======================================================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Copy ANTs binaries into SCT folder.

USAGE
  """+os.path.basename(__file__)+""" -f <folder_ants> -s <os_name>

MANDATORY ARGUMENTS
  -f <folder_ants>      ANTs folder. 
  -s {osx,linux}        name of the OS.

EXAMPLE
  """+os.path.basename(__file__)+""" -f ~/antsbin/bin -s osx\n"""

    sys.exit(2)


# main
#=======================================================================================================================

folder_bin = 'bin'
prefix_sct = 'sct_'
listOS = ['osx', 'linux']
list_file_ants = ['antsApplyTransforms', 'antsRegistration', 'antsSliceRegularizedRegistration', 'ANTSUseLandmarkImagesToGetAffineTransform', 'ANTSUseLandmarkImagesToGetBSplineDisplacementField', 'ComposeMultiTransform', 'ImageMath', 'ThresholdImage']
OSname = ''


# Check input param
try:
    opts, args = getopt.getopt(sys.argv[1:],'hf:s:')
except getopt.GetoptError as err:
    print str(err)
    usage()
if not opts:
    usage()
for opt, arg in opts:
    if opt == '-h':
        usage()
    elif opt in ('-s'):
        OSname = str(arg)
    elif opt in ('-f'):
        folder_ants = str(arg)

# check if OS exists
if OSname not in listOS:
    sct.printv('\nERROR: OS name should be one of the following: '+'[%s]' % ', '.join(map(str, listOS))+'\n', 1, 'error')

# check if ANTs folder exists
if not os.path.exists(folder_ants):
    sct.printv('\nERROR: Path '+folder_ants+' does not exist.\n', 1, 'error')

# build destination path name
path_bin = os.path.join(path_sct, folder_bin, OSname)

# loop across ANTs files
for file_ants in list_file_ants:
    # check if file exists
    if not os.path.exists(os.path.join(folder_ants, file_ants)):
        sct.printv('\nERROR: File '+folder_ants+file_ants+' does not exist.\n', 1, 'error')
    # copy data
    shutil.copy(os.path.join(folder_ants, file_ants), os.path.join(path_bin, prefix_sct + file_ants)

print "done!\n"

