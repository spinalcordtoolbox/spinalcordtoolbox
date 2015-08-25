#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create package with appropriate version number.
#
# Author: Julien Cohen-Adad, Benjamin De Leener
#

# TODO: remove quick fix with folder_sct_temp

import os
import getopt
import sys
from numpy import loadtxt
sys.path.append('../scripts')
import sct_utils as sct

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print 'USAGE: \n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Create a package of the Spinal Cord Toolbox.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+'\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -s <OS name>      name of the OS {osx,linux}.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  create_package.py -s linux\n'
    sys.exit(2)

listOS = ['osx', 'linux']
OSname = ''
# Check input param
try:
    opts, args = getopt.getopt(sys.argv[1:],'hs:')
except getopt.GetoptError as err:
    print str(err)
    usage()
for opt, arg in opts:
    if opt == '-h':
        usage()
    elif opt in ('-s'):
        OSname = str(arg)

if OSname not in listOS:
    print 'ERROR: OS name should be one of the following: '+'[%s]' % ', '.join(map(str,listOS))
    usage()

# get version
with open ("../version.txt", "r") as myfile:
    version = myfile.read().replace('\n', '')

# create output folder
folder_sct = '../spinalcordtoolbox_v'+version+'_'+OSname+'/'
if os.path.exists(folder_sct):
    sct.run('rm -rf '+folder_sct)
sct.run('mkdir '+folder_sct)

# copy folders
sct.run('mkdir '+folder_sct+'spinalcordtoolbox')
sct.run('cp installer.py '+folder_sct)
sct.run('cp ../README.md '+folder_sct+'spinalcordtoolbox/')
sct.run('cp ../LICENSE '+folder_sct+'spinalcordtoolbox/')
sct.run('cp ../version.txt '+folder_sct+'spinalcordtoolbox/')
sct.run('cp ../batch_processing.sh '+folder_sct+'spinalcordtoolbox/')
sct.run('cp ../batch_processing.sh '+folder_sct)
sct.run('cp -r ../flirtsch '+folder_sct+'spinalcordtoolbox/')
sct.run('cp -r ../scripts '+folder_sct+'spinalcordtoolbox/')

# install
sct.run('mkdir '+folder_sct+'spinalcordtoolbox/install')
sct.run('cp -r ../install/create_links.sh '+folder_sct + 'spinalcordtoolbox/install/')
sct.run('cp -r ../install/requirements ' + folder_sct + 'spinalcordtoolbox/install/')

# bin
sct.run('mkdir '+folder_sct+'spinalcordtoolbox/bin')
if OSname == 'osx':
    sct.run('cp -r ../bin/osx/* '+folder_sct+'spinalcordtoolbox/bin/')
elif OSname == 'linux':
    sct.run('cp -r ../bin/linux/* '+folder_sct+'spinalcordtoolbox/bin/')

# data
sct.run('cp -rf ../data '+folder_sct+'spinalcordtoolbox/')

# testing
sct.run('mkdir '+folder_sct+'spinalcordtoolbox/testing')
sct.run('cp ../testing/*.py '+folder_sct+'spinalcordtoolbox/testing/')

# remove .DS_Store files
sct.run('find '+folder_sct+' -type f -name .DS_Store -delete')

# remove AppleDouble files - doesn't work on Linux
if OSname == 'osx':
    sct.run('find '+folder_sct+' -type d | xargs dot_clean -m')

# remove python-related files
sct.run('rm '+folder_sct+'spinalcordtoolbox/scripts/*.pyc')
if os.path.exists(folder_sct+'spinalcordtoolbox/scripts/.idea'):
    sct.run('rm -rf '+folder_sct+'spinalcordtoolbox/scripts/.idea')

# go to parent directory to be able to tar without de-tarring error in mac OSX
os.chdir('../')

# compress folder
folder_sct_temp = 'spinalcordtoolbox_v'+version+'_'+OSname+'/'

sct.run('tar -cvzf spinalcordtoolbox_v'+version+'_'+OSname+'.tar.gz '+folder_sct_temp)

print "done!\n"


