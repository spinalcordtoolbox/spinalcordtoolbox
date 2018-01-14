#!/usr/bin/env python
#########################################################################################
#
# Install patch.
#
# Note for the developer:
# =======================
# Patch are created manually as follows:
# - get the previous patch version from sourceforge (if it exists)
# - rename it (or if not exist, create) a folder named: 1.0.2 (replace with appropriate patch version)
# - copy required files following the same folder organization. Example:
#     spinalcordtoolbox_patch_1.0.2/scripts/sct_dmri_moco.py
#     spinalcordtoolbox_patch_1.0.2/version.txt
#     spinalcordtoolbox_patch_1.0.2/install_patch.py
#   Each patch must contains all the previous patches (e.g., patch_1.0.3 must contains files from patch_1.0.1 and patch_1.0.2)
# - create zip file (easier than tar.gz) and rename the file patch_1.0.2
# - upload patch on sourceforge and github
# - update patches/patches.txt file with the new patch version
#
# The user install the patch by:
# - unzipping the file (double click)
# - launching "./install_patch.py"
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Benjamin De Leener
# Created: 2014-07-03
# Modified: 2015-01-27
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: make a script that creates patch files.


import sys, io, os, getopt
import signal

# small function for input with timeout
def interrupted(signum, frame):
    """called when read times out"""
    print 'interrupted!'
signal.signal(signal.SIGALRM, interrupted)

def input_timeout(text):
    try:
        foo = raw_input(text)
        return foo
    except:
        # timeout
        return

# get path of the toolbox to be able to import sct_utils
path_sct = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(path_sct, 'scripts'))
import sct_utils as sct
from sct_utils import UnsupportedOs, Os, Version, MsgUser

# main
#=======================================================================================================================
def main():
    issudo = "sudo "

    # check if user is sudoer
    if os.geteuid() == 0:
        print "Sorry, you are root. Please type: ./installer without sudo. Your password will be required later. Exit program\n"
        sys.exit(2)

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'ha')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-a'):
            issudo = ""

    try:
        this_computer = Os()
    except UnsupportedOs, e:
        MsgUser.debug(str(e))
        raise InstallFailed(str(e))

    # fetch version of the toolbox
    print 'Fetch version of the toolbox... '
    with open(os.path.join(path_sct, "version.txt"), "r") as myfile:
        version_sct = Version(myfile.read().rstrip())
    print "  toolbox version: "+str(version_sct)

    # fetch version of the patch
    print 'Fetch version of the patch... '
    with open("version.txt", "r") as myfile:
        version_patch = Version(myfile.read().rstrip())
    print "  patch version: "+str(version_patch)

    # if patch is not compatible with this release, send message and quit.
    print 'Check compatibility... '
    if version_sct >= version_patch:
        MsgUser.warning("This patch is not newer than the current version. Are you sure you want to install it?")
        install_new = ""
        signal.alarm(120)
        while install_new not in ["yes", "no"]:
            install_new = input_timeout("[yes|no]: ")
        signal.alarm(0)
        if install_new == "no":
            sys.exit(2)
    elif not version_sct.isEqualTo_MajorMinor(version_patch):
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
            # check if file is a bin/ file. If not, copy the old way. If so, copy bin in the right folder
            file_src = os.path.join(path_name, file_name+ext_name)
            if "bin/" not in path_name:
                file_dest = os.path.join(path_sct, path_name[1:], file_name+ext_name)
                # if destination folder does no exist, create it
                if not os.path.exists(os.path.join(path_sct, path_name[1:])):
                    sct.run(issudo+'mkdir '+ os.path.join(path_sct, path_name[1:]))
                # copy file
                sct.run(issudo+'cp '+file_src+' '+file_dest)
            else:
                if this_computer.os in path_name:
                    # path_name ends with .../bin/osx/ or .../bin/linux/ so we can get the parent directory
                    path_name_new = os.path.dirname(path_name[0:-1])
                    file_dest = os.path.join(path_sct, path_name_new[1:], file_name+ext_name)
                    # copy file
                    sct.run(issudo+'cp '+file_src+' '+file_dest)

    # re-create links
    print 'Update links...'
    sudo_links = ''
    if issudo == "":
        sudo_links = ' -a'
    status, output = sct.run('${SCT_DIR}/install/create_links.sh'+sudo_links)
    print output

    print "Done!\n"

class InstallFailed(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class InstallationResult(object):
    SUCCESS = 0
    WARN = 1
    ERROR = 2

    def __init__(self,result,status,message):
        self.result = result
        self.status = status
        self.message = message

    def __nonzero__(self):
        self.status


def usage():
    print """
""" + os.path.basename(__file__) + """
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
Install Spinal Cord Toolbox patch in $SCT_DIR

USAGE
""" + os.path.basename(__file__) + """ -p <path>

MANDATORY ARGUMENTS
-a                          allow for non-admin installation
-h                          display this help
  """

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    try:
        main()
    except InstallFailed, e:
        MsgUser.failed(e.value)
        exit(1)
    except UnsupportedOs, e:
        MsgUser.failed(e.value)
        exit(1)
    except KeyboardInterrupt, e:
        MsgUser.failed("Install aborted by the user.")
        exit(1)
