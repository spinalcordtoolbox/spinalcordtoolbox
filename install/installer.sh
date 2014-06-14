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

# check if .bashrc was already modified. If so, we delete lines about sct to be sure.
echo
echo "Edit .bashrc..."
if grep -q "SPINALCORDTOOLBOX" ~/.bashrc; then
  echo "Deleting previous sct entries in .bashrc"
  cmd="awk '!/SCT_DIR|SPINALCORDTOOLBOX/' ~/.bashrc > .bashrc_temp && > ~/.bashrc && cat .bashrc_temp >> ~/.bashrc && rm .bashrc_temp"
  echo ">> $cmd"
  awk '!/SCT_DIR|SPINALCORDTOOLBOX/' ~/.bashrc > .bashrc_temp && > ~/.bashrc && cat .bashrc_temp >> ~/.bashrc && rm .bashrc_temp
fi

# edit .bashrc. Add bin and libraries
echo '' >> ~/.bashrc
echo '# SPINALCORDTOOLBOX' >> ~/.bashrc
echo "SCT_DIR=\"${SCT_DIR}\"" >> ~/.bashrc
echo 'export PATH=${PATH}:$SCT_DIR/scripts' >> ~/.bashrc
echo 'export PATH=${PATH}:$SCT_DIR/bin' >> ~/.bashrc
unamestr=`uname`
if [[ ! "$unamestr" == 'Linux' ]]; then
  echo 'export DYLD_LIBRARY_PATH=${SCT_DIR}/lib:$DYLD_LIBRARY_PATH' >> ~/.bashrc
fi
echo 'export SCT_DIR PATH' >> ~/.bashrc

# check if .bash_profile exists. If so, we check if link to .bashrc is present in it. If not, we add it at the end.
if [ -e "$HOME/.bash_profile" ]; then
  if grep -q "source ~/.bashrc" ~/.bash_profile; then
    echo
    echo ".bashrc correctly called in .bash_profile"
  else
    echo
    echo "edit .bash_profile..."
    echo "if [ -f ~/.bashrc ]; then" >> ~/.bash_profile
    echo '  source ~/.bashrc' >> ~/.bash_profile
    echo 'fi' >> ~/.bash_profile
  fi
fi

# launch .bash_profile. This line doesn't always work. Best way is to open a new terminal.
. ~/.bashrc

# check if other dependent software are installed
echo
echo "Check if other dependent software are installed..."
cmd="python ${SCT_DIR}/scripts/sct_check_dependences.py"
echo ">> $cmd"; $cmd

# display stuff
echo
echo "---"
echo "Done! If no error appeared above, you can delete this folder by typing:"
echo "> cd .."
echo "> rm -rf spinalcordtoolbox_v*"
echo
echo "To see all commands available, start a new Terminal and type \"sct\" then backslash"
echo 
echo "To get started, go the created folder by typing:"
echo "> cd sct_testing"
echo "and follow instructions here: https://sourceforge.net/p/spinalcordtoolbox/wiki/get_started/"
echo "Enjoy :-)"
echo "---"
echo
