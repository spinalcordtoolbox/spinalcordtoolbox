#!/usr/bin/env python
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

class Installer:
	def __init__(self):
		self.path_install = "/usr/local"
		self.issudo = "sudo "

		# check if user is sudoer
		status, output = commands.getstatusoutput(cmd)
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
		if path[-1:] == '/':
            path = path[:-1]

        self.SCT_DIR = self.path_install+"/spinalcordtoolbox"


		# check if SCT folder already exists - if so, delete it
		print ""
		print "Check if spinalcordtoolbox is already installed (if so, delete it)..."
		if os.path.isdir(self.SCT_DIR):
			cmd="${ISSUDO}rm -rf ${SCT_DIR}"
			print ">> "+cmd
			status, output = commands.getstatusoutput(cmd)
    		if status != 0:
        		print '\nERROR! \n'+output+'\nExit program.\n'
        		sys.exit(2)

        # create SCT folder
        print "Create folder: "+self.SCT_DIR+" ..."
        os.makedirs(self.SCT_DIR)
        print "OK!"

        # copy SCT files
        print "\nCopy Spinal Cord Toolbox on your computer..."
        cmd = self.issudo+"cp -r spinalcordtoolbox/* "+self.SCT_DIR
        print ">> "+cmd
        status, output = commands.getstatusoutput(cmd)
    	if status != 0:
        	print '\nERROR! \n'+output+'\nExit program.\n'


		# check if .bashrc was already modified. If so, we delete lines related to SCT
		print "\nEdit .bashrc..."
		if "SPINALCORDTOOLBOX" in open('~/.bashrc').read():
			print "Deleting previous sct entries in .bashrc"
		  	cmd = "awk '!/SCT_DIR|SPINALCORDTOOLBOX/' ~/.bashrc > .bashrc_temp && > ~/.bashrc && cat .bashrc_temp >> ~/.bashrc && rm .bashrc_temp"
		  	print ">> "+cmd
		  	status, output = commands.getstatusoutput(cmd)
    		if status != 0:
        		print '\nERROR! \n'+output+'\nExit program.\n'

        # check if .bashrc exist. If not, create it.
        if not os.path.isfile("~/.bashrc"):
        	with open("~/.bashrc", "a") as bashrc:
        		bashrc.write("")
        		bashrc.close()

		# edit .bashrc. Add bin
		with open("~/.bashrc", "a") as bashrc:
    		bashrc.write("# SPINALCORDTOOLBOX (added on "+date.today()+")")
    		bashrc.write("SCT_DIR=\""+self.SCT_DIR+"\"")
    		bashrc.write("export PATH=${PATH}:$SCT_DIR/bin")
    		# add PYTHONPATH variable to allow import of modules
    		bashrc.write("export PYTHONPATH=${PYTHONPATH}:$SCT_DIR/scripts")
    		bashrc.write("export SCT_DIR PATH")
    		# forbid to run several ITK instances in parallel (see issue #201).
    		bashrc.write("export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1")
    		bashrc.close()

		# check if .bash_profile exists. If so, we check if link to .bashrc is present in it. If not, we add it at the end.
		if os.path.isfile("~/.bash_profile"):
			if "source ~/.bashrc" in open('~/.bash_profile').read():
				print "\n.bashrc seems to be called in .bash_profile"
				# TODO: check for the case if the user did comment source ~/.bashrc in his .bash_profile
			else:
				print "edit .bash_profile..."
				with open("~/.bash_profile", "a") as bashprofile:
					bashprofile.write("if [ -f ~/.bashrc ]; then")
					bashprofile.write("  source ~/.bashrc")
					bashprofile.write("fi")
					bashprofile.close()

		# launch .bashrc. This line doesn't always work. Best way is to open a new terminal.
		cmd = ". ~/.bashrc"
		print ">> "+cmd
		status, output = commands.getstatusoutput(cmd)
    	if status != 0:
        	print '\nERROR! \n'+output+'\nExit program.\n'
		


		# install required software
		print "\nInstall required software... "
		cmd = "./requirements/requirements.sh"
		print ">> "+cmd
		status, output = commands.getstatusoutput(cmd)
    	if status != 0:
        	print '\nERROR! \n'+output+'\nExit program.\n'

		# Checking if patches are available. If so, install them.


		# Create links to python scripts
		print "\nCreate links to python scripts..."
		cmd = self.SCT_DIR+"/install/create_links.sh"
		print ">> "+cmd
		status, output = commands.getstatusoutput(cmd)
    	if status != 0:
        	print '\nERROR! \n'+output+'\nExit program.\n'

		# check if other dependent software are installed
		print "\nCheck if other dependent software are installed..."
		cmd="sct_check_dependences"
		print ">> "+cmd
		status, output = commands.getstatusoutput(cmd)
    	if status != 0:
        	print '\nERROR! \n'+output+'\nExit program.\n'
        else:
        	print output

		# display stuff
		print """\n"========================================================================================"
		Done! If you had errors, please start a new Terminal and run the following command:
		> sct_check_dependences
		
		If you are still getting errors, please post an issue here: https://sourceforge.net/p/spinalcordtoolbox/discussion/help/
		or contact the developers.
		
		You can now delete this folder by typing:
		> cd ..
		> rm -rf """+os.getcwd()+"""

		To get started, open a new Terminal and follow instructions here: https://sourceforge.net/p/spinalcordtoolbox/wiki/get_started/
		"""


def usage():
	print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
Install the Spinal Cord Toolbox

USAGE
"""+os.path.basename(__file__)+""" -p <path>

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
    main()
