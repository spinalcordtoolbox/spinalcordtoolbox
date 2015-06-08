#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Script to create links for FSL when filenames starts with FSL5.0-
#
# Author: Benjamin De Leener
# Modified: 2014-09-03

import os
import getopt
import sys
import commands
sys.path.append('../scripts')
import sct_utils as sct


def usage():
    print 'USAGE: \n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Create for FSL in the Spinal Cord Toolbox.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+'\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -a      remove admin rights.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  create_package.py -a\n'
    sys.exit(2)

issudo = 'sudo '
path_fsl = ''
prefix_fsl = ''
# Check input param
try:
    opts, args = getopt.getopt(sys.argv[1:], 'ha:f:p:')
except getopt.GetoptError as err:
    print str(err)
    usage()
for opt, arg in opts:
    if opt == '-h':
        usage()
    elif opt in '-a':
        issudo = ''
    elif opt in '-f':
        path_fsl = arg
    elif opt in '-p':
        prefix_fsl = arg

# get path of SCT
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

# create soft link to each script in SCT_DIR/script
if prefix_fsl is '':
    prefix_fsl = 'fsl5.0-'
print "Creating soft links for each fsl script that start with ", prefix_fsl

if path_fsl is '':
    status, path_fsl = commands.getstatusoutput('which ' + prefix_fsl + 'fslmaths')
    path_fsl = os.path.dirname(path_fsl)
if path_fsl is '':
    print 'ERROR: the FSL folder was not found! Please provide it using -f option.'

for fsl_script in os.listdir(path_fsl):
    if fsl_script.startswith(prefix_fsl):
        script_name = fsl_script[len(prefix_fsl):]
        sct.run(issudo + 'ln -sf ' + path_fsl+'/'+fsl_script + ' ' + path_sct+'/bin/'+script_name)

print "Done!"
