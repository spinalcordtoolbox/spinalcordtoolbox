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
        '\n'\
    # sys.exit(2)

# listOS = ['osx', 'linux']
# OSname = ''
# Check input param
try:
    opts, args = getopt.getopt(sys.argv[1:],'h')
except getopt.GetoptError as err:
    print str(err)
    usage()
for opt, arg in opts:
    if opt == '-h':
        usage()
    # elif opt in ('-s'):
    #     OSname = str(arg)

# if OSname not in listOS:
#     print 'ERROR: OS name should be one of the following: '+'[%s]' % ', '.join(map(str,listOS))
#     usage()

# get version
with open ("../version.txt", "r") as myfile:
    version = myfile.read().replace('\n', '')

# create output folder
folder_sct = '../sct_v'+version+'/'
if os.path.exists(folder_sct):
    sct.run('rm -rf '+folder_sct)
sct.run('mkdir '+folder_sct)

# copy folders
sct.run('cp ../install_sct '+folder_sct)
sct.run('cp ../README.md '+folder_sct)
sct.run('cp ../LICENSE '+folder_sct)
sct.run('cp ../version.txt '+folder_sct)
sct.run('cp ../commit.txt '+folder_sct)
sct.run('cp ../batch_processing.sh '+folder_sct)
sct.run('cp ../batch_processing.sh '+folder_sct)
sct.run('cp -r ../scripts '+folder_sct)
sct.run('cp -r ../install '+folder_sct)
sct.run('cp -r ../testing '+folder_sct)
#sct.run('cp -r ../external '+folder_sct+'sct/')

# install
#sct.run('mkdir '+folder_sct+'install')
# sct.run('cp -r ../install/create_links.sh '+folder_sct + 'spinalcordtoolbox/install/')
# sct.run('cp -r ../install/install_external.py '+folder_sct + 'spinalcordtoolbox/install/')
#sct.run('cp -r ../install/requirements '+folder_sct+'install/')

# bin
#sct.run('mkdir '+folder_sct+'sct/bin')
#sct.run('cp -r ../bin '+folder_sct+'sct/')
# if OSname == 'osx':
#     sct.run('cp -r ../bin/osx/* '+folder_sct+'spinalcordtoolbox/bin/')
# elif OSname == 'linux':
#     sct.run('cp -r ../bin/linux/* '+folder_sct+'spinalcordtoolbox/bin/')

# data
#sct.run('cp -rf ../data '+folder_sct+'sct/')

# testing
#sct.run('mkdir '+folder_sct+'testing')
#sct.run('cp ../testing/*.py '+folder_sct+'testing/')

# remove .DS_Store files
sct.run('find '+folder_sct+' -type f -name .DS_Store -delete')
# remove Pycharm-related files
sct.run('find '+folder_sct+' -type f -name *.pyc -delete')
sct.run('find '+folder_sct+' -type f -name *.idea -delete')

# remove AppleDouble files - doesn't work on Linux
# if OSname == 'osx':
#     sct.run('find '+folder_sct+' -type d | xargs dot_clean -m')

# remove Pycharm-related files
#sct.run('rm '+folder_sct+'scripts/*.pyc')
#if os.path.exists(folder_sct+'scripts/.idea'):
#    sct.run('rm -rf '+folder_sct+'scripts/.idea')

# go to parent directory to be able to tar without de-tarring error in mac OSX
os.chdir('../')

# compress folder
folder_sct_temp = 'sct_v'+version+'/'
sct.run('tar -cvzf sct_v'+version+'.tar.gz '+folder_sct_temp)

# remove temp folder
sct.run('rm -rf sct_v'+version)

print "done!\n"
