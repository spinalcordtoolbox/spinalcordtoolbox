#!/usr/bin/env python
#
# Copy ANTs binaries into SCT folder. 
#
# Author: Julien Cohen-Adad
# modified: 2014-11-01

# TODO: remove quick fix with folder_sct_temp

import os
import getopt
import sys
import commands
sys.path.append('../scripts')
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

# get path of SCT
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

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
        folder_ants = sct.slash_at_the_end(str(arg), 1)  # add slash at the end

# check if OS exists
if OSname not in listOS:
    sct.printv('\nERROR: OS name should be one of the following: '+'[%s]' % ', '.join(map(str, listOS))+'\n', 1, 'error')

# check if ANTs folder exists
if not os.path.exists(folder_ants):
    sct.printv('\nERROR: Path '+folder_ants+' does not exist.\n', 1, 'error')

# build destination path name
path_bin = sct.slash_at_the_end(path_sct, 1)+folder_bin+'/'+OSname+'/'

# loop across ANTs files
for file_ants in list_file_ants:
    # check if file exists
    if not os.path.exists(folder_ants+file_ants):
        sct.printv('\nERROR: File '+folder_ants+file_ants+' does not exist.\n', 1, 'error')
    # copy data
    sct.run('cp '+folder_ants+file_ants+' '+path_bin+prefix_sct+file_ants, 1)

print "done!\n"

