#!/bin/bash
#
# Installer for spinal cord toolbox.
#
# This script will install the spinal cord toolbox under and configure your environment.
# Must be run as a non-administrator (no sudo).
# Installation location: /usr/local/spinalcordtoolbox/

# parameters
#PATH_INSTALL="/usr/local/"
SCT_DIR="/usr/local/spinalcordtoolbox"

echo

# check if user is sudoer
if [ "$(whoami)" == "root" ]; then
  echo "Sorry, you are root. Please type: ./installer without sudo. Your password will be required later."
  echo
  exit 1
fi

# check if folder already exists - if so, delete it
echo
echo "Check if spinalcordtoolbox is already installed (if so, delete it)..."
if [ -e "${SCT_DIR}" ]; then
  cmd="sudo rm -rf ${SCT_DIR}"
  echo ">> $cmd"; $cmd
fi

# create folder
echo
echo "Create folder: /usr/local/spinalcordtoolbox..."
cmd="sudo mkdir ${SCT_DIR}"
echo ">> $cmd"; $cmd

# copy files
echo
echo "Copy toolbox..."
cmd="sudo cp -r spinalcordtoolbox/* ${SCT_DIR}"
echo ">> $cmd"; $cmd

# copy testing files
echo
echo "Copy example data & scripts..."
if [ -e "../sct_testing" ]; then
  cmd="sudo rm -rf ../sct_testing"
  echo ">> $cmd"; $cmd
fi
cmd="sudo mkdir ../sct_testing"
echo ">> $cmd"; $cmd
cmd="sudo cp -r spinalcordtoolbox/testing/ ../sct_testing"
echo ">> $cmd"; $cmd
cmd="sudo chmod -R 775 ../sct_testing"
echo ">> $cmd"; $cmd

# edit bash_profile
# check if .bash_profile was already modified
echo
echo "Edit .bash_profile..."
if grep -q "SPINALCORDTOOLBOX" ~/.bash_profile; then
  echo "  .bash_profile was already modified previously."
else
  echo '' >> ~/.bash_profile
  echo '# SPINALCORDTOOLBOX' >> ~/.bash_profile
  echo "SCT_DIR=\"${SCT_DIR}\"" >> ~/.bash_profile
  echo 'export PATH=${PATH}:$SCT_DIR/scripts' >> ~/.bash_profile
  echo 'export PATH=${PATH}:$SCT_DIR/bin' >> ~/.bash_profile
  unamestr=`uname`
  if [[ "$unamestr" == 'Linux' ]]; then
    echo 'export LD_LIBRARY_PATH=${SCT_DIR}/lib:$LD_LIBRARY_PATH' >> ~/.bash_profile
  else
    echo 'export DYLD_LIBRARY_PATH=${SCT_DIR}/lib:$DYLD_LIBRARY_PATH' >> ~/.bash_profile
  fi
  echo 'export SCT_DIR PATH' >> ~/.bash_profile
fi

# check if other dependent software are installed
echo
echo "Check if other dependent software are installed..."
cmd="python ${SCT_DIR}/scripts/sct_check_dependences.py"
echo ">> $cmd"; $cmd

# display stuff
echo
echo "---"
echo "Done! If no error appeared above, you can delete this folder."
echo "To see all commands available, type \"sct\" then backslash"
echo "To get more info about the toolbox, please see /usr/local/spinalcordtoolbox/README.txt"
echo "To get started, look at the created folder: \"sct_testing\""
echo "Please send your comments here: http://sourceforge.net/p/spinalcordtoolbox/discussion/"
echo "Enjoy :-)"
echo "---"
echo
