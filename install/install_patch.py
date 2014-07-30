#!/usr/bin/env python
#########################################################################################
#
# Install patch.
#
# Note for the developer:
# =======================
# Patch are created manually as follows:
# - get the previous patch version from sourceforge (if it exists)
# - rename it (or if not exist, create) a folder named: spinalcordtoolbox_patch_1.0.2 (replace with appropriate patch version)
# - copy required files following the same folder organization. Example:
#     spinalcordtoolbox_patch_1.0.2/scripts/sct_dmri_moco.py
#     spinalcordtoolbox_patch_1.0.2/version.txt
#     spinalcordtoolbox_patch_1.0.2/install_patch.py
# - create zip file (easier than tar.gz)
# - upload patch on sourceforge
#
# The user install the patch by:
# - unzipping the file (double click)
# - launching "./install_patch.py"
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: make a script that creates patch files.


import sys
import commands
import os
import platform

# get path of the toolbox to be able to import sct_utils
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
sys.path.append(path_sct+'/scripts')
import sct_utils as sct


# main
#=======================================================================================================================
def main():

    # initialization
    os_running = 'not identified'

    print

    # check OS
    print 'Check which OS is running... '
    platform_running = sys.platform
    if (platform_running.find('darwin') != -1):
        os_running = 'osx'
    elif (platform_running.find('linux') != -1):
        os_running = 'linux'
    print '  '+os_running+' ('+platform.platform()+')'

    # fetch version of the toolbox
    print 'Fetch version of the toolbox... '
    with open (path_sct+"/version.txt", "r") as myfile:
        version_sct = myfile.read().replace('\n', '')
    print "  toolbox version: "+version_sct

    # fetch version of the patch
    print 'Fetch version of the patch... '
    with open ("version.txt", "r") as myfile:
        version_patch = myfile.read().replace('\n', '')
    print "  patch version: "+version_patch

    # if patch is not compatible with this release, send message and quit.
    print 'Check compatibility... '
    version_sct_num = version_sct.split('.')
    version_patch_num = version_patch.split('.')
    if not ( ( version_sct_num[0] == version_patch_num[0] ) and ( version_sct_num[1] == version_patch_num[1] ) ):
        print "  ERROR: Patch is not compatible with this release. Patch version X.Y.Z should correspond to release" \
                "  version X.Y. Exit program.\n"
        sys.exit(2)
    else:
        print "  OK"

    # list all files in patch
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames]

    # copy files one by one (to inform user)
    for f in files:
        path_name, file_name, ext_name = sct.extract_fname(f)

        # check if .DS_Store (could happen during package creation)
        if not file_name == ".DS_Store":
            # copy file
            # print path_name[2:]+' ++ '+file_name+' ++ '+ext_name
            file_src = path_name+file_name+ext_name
            file_dest = path_sct+path_name[1:]+file_name+ext_name
            sct.run('sudo cp '+file_src+' '+file_dest)

    print "Done!\n"


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()