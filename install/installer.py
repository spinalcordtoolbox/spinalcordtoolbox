#!/usr/bin/env python
# coding=utf-8
#########################################################################################
#
# Installer for spinal cord toolbox.
# 
# This script will install the spinal cord toolbox under and configure your environment.
# Must be run as a non-administrator (no sudo).
# Installation location: /usr/local/spinalcordtoolbox/
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad, Benjamin De Leener
# Modified: 2015-01-23
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import os
import sys
import commands
import getopt
from datetime import date
import utllib2
import platform


class Installer:
    def __init__(self):
        self.path_install = "/usr/local"
        self.issudo = "sudo "

        # check if user is sudoer
        if os.geteuid() == 0:
            print "Sorry, you are root. Please type: ./installer without sudo. Your password will be required later. Exit program\n"
            sys.exit(2)

        # Check input parameters

        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hp:')
        except getopt.GetoptError:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-p'):
                self.path_install = arg
            self.issudo = ""

        print ""
        print "============================="
        print "SPINAL CORD TOOLBOX INSTALLER"
        print "============================="

        if not os.path.isdir(self.path_install):
            print "ERROR: The path you entered does not exist: ${PATH_INSTALL}. Create it first. Exit program\n"
            sys.exit(2)

        # check if last character is "/". If so, remove it.
        if self.path_install[-1:] == '/':
            self.path_install = self.path_install[:-1]

        self.SCT_DIR = self.path_install + "/spinalcordtoolbox"

        # Retrieving home folder because in python, paths with ~ do not seem to work.
        self.home = os.path.expanduser('~')

        # check if SCT folder already exists - if so, delete it
        print ""
        print "Check if spinalcordtoolbox is already installed (if so, delete it)..."
        if os.path.isdir(self.SCT_DIR):
            cmd = self.issudo+"rm -rf "+self.SCT_DIR
            print ">> " + cmd
            status, output = commands.getstatusoutput(cmd)
            if status != 0:
                print 'ERROR! \n' + output + '\nExit program.\n'
                sys.exit(2)

        # create SCT folder
        print "Create folder: " + self.SCT_DIR + " ..."
        cmd = self.issudo+"mkdir "+self.SCT_DIR
        print ">> " + cmd
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print output + '\n'

        # Checking if a new version of the toolbox is available. If so, change it.
        # Check the version on GitHub Master branch. If a new release is available, ask the user if he want to install it.
        # fetch version of the toolbox
        print 'Fetch version of the Spinal Cord Toolbox... '
        with open ("spinalcordtoolbox/version.txt", "r") as myfile:
            version_sct = myfile.read().replace('\n', '')
        print "  version: "+version_sct
        version_sct_split = version_sct.split('.')

        # fetch version of the toolbox online
        url_version = "http://github.com/neuropoly/spinalcordtoolbox/blob/master/version.txt"
        file_name = url_version.split('/')[-1]
        u = urllib2.urlopen(url_version)
        f = open(file_name, 'w')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print "Downloading: %s Bytes: %s" % (file_name, file_size)
        with open (file_name, "r") as myfile:
            version_sct_online = myfile.read().replace('\n', '')
        print "  latest available version: "+version_sct_online
        version_sct_online_split = version_sct_online.split('.')

        if version_sct_split[0] != version_sct_online_split[0] or version_sct_split[1] != version_sct_online_split[1]:
            print "Warning: A new version of the Spinal Cord Toolbox is available online. Do you want to install it?"
            install_new = ""
            while install_new not in ["yes","no"]:
                install_new = input("[yes|no]: ")
            if install_new == "yes":
                print "The automatic installation of a new release or version of the toolbox is not supported yet. Please download it on https://sourceforge.net/projects/spinalcordtoolbox/"

        # copy SCT files
        print "\nCopy Spinal Cord Toolbox on your computer..."
        cmd = self.issudo + "cp -r spinalcordtoolbox/* " + self.SCT_DIR
        print ">> " + cmd
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'


        # check if .bashrc was already modified. If so, we delete lines related to SCT
        print "\nEdit .bashrc..."

        # check if .bashrc exist. If not, create it.
        if not os.path.isfile(self.home+"/.bashrc"):
            print "  ~/.bashrc does not exist. Creating it..."
            open(self.home+'/.bashrc', 'w+').close()
        else:
            if "SPINALCORDTOOLBOX" in open(self.home+'/.bashrc').read():
                print "  Deleting previous SCT entries in .bashrc"
                cmd = "awk '!/SCT_DIR|SPINALCORDTOOLBOX/' ~/.bashrc > .bashrc_temp && > ~/.bashrc && cat .bashrc_temp >> ~/.bashrc && rm .bashrc_temp"
                print ">> " + cmd
                status, output = commands.getstatusoutput(cmd)
                if status != 0:
                    print '\nERROR! \n' + output + '\nExit program.\n'

        # edit .bashrc. Add bin
        with open(self.home+"/.bashrc", "a") as bashrc:
            bashrc.write("\n# SPINALCORDTOOLBOX (added on " + str(date.today()) + ")")
            bashrc.write("\nSCT_DIR=\"" + self.SCT_DIR + "\"")
            bashrc.write("\nexport PATH=${PATH}:$SCT_DIR/bin")
            # add PYTHONPATH variable to allow import of modules
            bashrc.write("\nexport PYTHONPATH=${PYTHONPATH}:$SCT_DIR/scripts")
            bashrc.write("\nexport SCT_DIR PATH")
            # forbid to run several ITK instances in parallel (see issue #201).
            bashrc.write("\nexport ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1")
            bashrc.close()

        # Because python script cannot source bashrc or bash_profile, it is necessary to modify environment in the current instance of bash
        os.environ['SCT_DIR'] = self.SCT_DIR
        os.environ['PATH'] = os.environ['PATH']+":"+self.SCT_DIR+"/bin"
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+":"+self.SCT_DIR+"/scripts"
        else:
            os.environ['PYTHONPATH'] = self.SCT_DIR+"/scripts"

        # check if .bash_profile exists. If so, we check if link to .bashrc is present in it. If not, we add it at the end.
        if os.path.isfile(self.home+"/.bash_profile"):
            if "source ~/.bashrc" in open(self.home+'/.bash_profile').read():
                print "\n.bashrc seems to be called in .bash_profile"
            # TODO: check for the case if the user did comment source ~/.bashrc in his .bash_profile
            else:
                print "edit .bash_profile..."
                with open(self.home+"/.bash_profile", "a") as bashprofile:
                    bashprofile.write("\nif [ -f ~/.bashrc ]; then")
                    bashprofile.write("\n  source ~/.bashrc")
                    bashprofile.write("\nfi")
                    bashprofile.close()

        # launch .bashrc. This line doesn't always work. Best way is to open a new terminal.
        cmd = ". ~/.bashrc"
        print ">> " + cmd
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'

        # install required software
        print "\nInstall required software... "
        os.chdir("requirements")
        cmd = "./requirements.sh"
        print ">> " + cmd
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'
        os.chdir("..")

        # Create links to python scripts
        print "\nCreate links to python scripts..."
        cmd = self.SCT_DIR+"/install/create_links.sh"
        if self.issudo is "":
            cmd = cmd+" -a"
        print ">> " + cmd
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'

        # Checking if patches are available. If so, install them. Patches installation is available from release 1.1
        if version_sct_split[0] == version_sct_online_split[0] and version_sct_split[1] == version_sct_online_split[1] and version_sct_split[2] != version_sct_online_split[2]:
            # check if a new release is available
            url_versions = "http://github.com/neuropoly/spinalcordtoolbox/blob/master/versions.txt"
            file_name = url_versions.split('/')[-1]
            u = urllib2.urlopen(url_versions)
            f = open(file_name, 'w')
            meta = u.info()
            file_size = int(meta.getheaders("Content-Length")[0])
            print "Downloading: %s Bytes: %s" % (file_name, file_size)
            versions_old = list
            with open (file_name, "r") as versions_file:
                versions_old.append(versions_file.readlines())
            for version_sct in versions_old:
                ver = version_sct.split('.')
                # As the versions are ordered from the newest to the latest, the first one with [0] and [1] term that will be found is either a new one or the same that is installed.
                if ver[0] == version_sct_split[0] and ver[1] == version_sct_split[1] and ver[2] != version_sct_split[2]:
                    ver_patch_split = ver[2].split('-')
                    # check if the patch is for linux, osx or both. If the patch is for both os, we install it. If not, we check.
                    install_patch = True
                    name_folder_patch = ver[2]
                    if len(ver_patch_split) > 1:
                        # check which platform is running
                        platform_running = sys.platform
                        if (platform_running.find('darwin') != -1):
                            os_running = 'osx'
                        elif (platform_running.find('linux') != -1):
                            os_running = 'linux'
                        if ver_patch_split[1] != os_running:
                            install_patch = False
                        else:
                            name_folder_patch = ver_patch_split[0] # keep only folder name, without os

                    # If a patch needs to be installed, install it.
                    if install_patch:
                        url_patch = "http://github.com/neuropoly/spinalcordtoolbox/blob/master/patches/patch_"+version_sct+".zip"
                        file_name = url_patch.split('/')[-1]
                        u = urllib2.urlopen(url_patch)
                        f = open(file_name, 'w')
                        meta = u.info()
                        file_size = int(meta.getheaders("Content-Length")[0])
                        print "Downloading: %s Bytes: %s" % (file_name, file_size)

                        # unzip patch
                        cmd = "unzip "+file_name
                        print ">> " + cmd
                        status, output = commands.getstatusoutput(cmd)
                        if status != 0:
                            print '\nERROR! \n' + output + '\nExit program.\n'

                        # launch patch installation
                        cmd = "python "+name_folder_patch+"/install_patch.py"
                        print ">> " + cmd
                        status, output = commands.getstatusoutput(cmd)
                        if status != 0:
                            print '\nERROR! \n' + output + '\nExit program.\n'

                        # Once the patch is installed, we do not check for other patches
                        break
                else:
                    # No new patch
                    break


        # check if other dependent software are installed
        print "\nCheck if other dependent software are installed..."
        cmd = "sct_check_dependences"
        print ">> " + cmd
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'
        else:
            print output

        # display stuff
        print """\n"========================================================================================"
Installation done! You may need to run a new Terminal in order to set environment variables.
If you had errors, please start a new Terminal and run the following command:
> sct_check_dependences -c -l

If you are still getting errors, please post an issue here: https://sourceforge.net/p/spinalcordtoolbox/discussion/help/
or contact the developers.

You can now delete this folder by typing:
> cd ..
> rm -rf """ + os.getcwd() + """

To get started, open a new Terminal and follow instructions here: https://sourceforge.net/p/spinalcordtoolbox/wiki/get_started/
"""


def usage():
    print """
""" + os.path.basename(__file__) + """
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
Install the Spinal Cord Toolbox

USAGE
""" + os.path.basename(__file__) + """ -p <path>

MANDATORY ARGUMENTS
-p <path>                   installation path. Do not add "/" at the end!
-h                          display this help
  """

    # exit program
    sys.exit(2)

# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    Installer()
