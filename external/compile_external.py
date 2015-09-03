#!/usr/bin/env python
#
# Compile external packages.
#

# TODO: before adding line for PYTHONPATH, check if already exists

import os
import commands

status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
path_denoise = "external/denoise/ornlm"

# go to folder
os.chdir(path_sct+'/'+path_denoise)

# compile
status, output = commands.getstatusoutput('python setup.py build_ext --inplace')
if not status:
    print output

# Retrieving home folder because in python, paths with ~ do not seem to work.
path_home = os.path.expanduser('~')

# REMOVED THE LINE BELOW: BETTER TO APPEND PYTHONPATH IN SCRIPT RATHER THAN POLLUTING BASHRC
# # add to .bashrc
# with open(path_home+"/.bashrc", "a") as bashrc:
#     bashrc.write("export PYTHONPATH=${PYTHONPATH}:${SCT_DIR}/"+path_denoise+"\n")
#     bashrc.close()

# # put in python environment for subsequent tests during installation
# if 'PYTHONPATH' in os.environ:
#     os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+":"+path_sct+path_denoise
# else:
#     os.environ['PYTHONPATH'] = path_sct+path_denoise

# source .bashrc
# !! This does not work, as python script launched a new process. Solutions are welcome!
# status, output = commands.getstatusoutput("source "+path_home+"/.bashrc")

print "Done! Open a new Terminal window to load environment variables."
