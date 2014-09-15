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
echo "============================="
echo "SPINAL CORD TOOLBOX INSTALLER"
echo "============================="

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
cmd="mv spinalcordtoolbox/testing ../sct_testing"
echo ">> $cmd"; $cmd
cmd="sudo chmod -R 775 ../sct_testing"
echo ">> $cmd"; $cmd

# remove testing in installation folder
echo
echo "Remove testing in installation folder"
cmd="sudo rm -rf ${SCT_DIR}/testing"
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

# edit .bashrc. Add bin
echo '' >> ~/.bashrc
echo "# SPINALCORDTOOLBOX (added on $(date +%Y-%m-%d))" >> ~/.bashrc
echo "SCT_DIR=\"${SCT_DIR}\"" >> ~/.bashrc
# echo 'export PATH=${PATH}:$SCT_DIR/scripts' >> ~/.bashrc # to remove
echo 'export PATH=${PATH}:$SCT_DIR/bin' >> ~/.bashrc
# add PYTHONPATH variable to allow import of modules
echo 'export PYTHONPATH=$SCT_DIR/scripts' >> ~/.bashrc

echo ${SCT_DIR}

unamestr='uname'
#if [[ ! "$unamestr" == 'Linux' ]]; then
#  echo 'export DYLD_LIBRARY_PATH=${SCT_DIR}/lib:$DYLD_LIBRARY_PATH' >> ~/.bashrc
#fi
echo 'export SCT_DIR PATH' >> ~/.bashrc

# check if .bash_profile exists. If so, we check if link to .bashrc is present in it. If not, we add it at the end.
if [ -e "$HOME/.bash_profile" ]; then
  if grep -q "source ~/.bashrc" ~/.bash_profile; then
    echo
    echo ".bashrc seems to be called in .bash_profile"
	# TOOD: check for the case if the user did comment source ~/.bashrc in his .bash_profile 
  else
    echo
    echo "edit .bash_profile..."
    echo "if [ -f ~/.bashrc ]; then" >> ~/.bash_profile
    echo '  source ~/.bashrc' >> ~/.bash_profile
    echo 'fi' >> ~/.bash_profile
  fi
fi

# launch .bashrc. This line doesn't always work. Best way is to open a new terminal.
. ~/.bashrc

# install required software
echo
echo " Install required software... "
cd requirements
cmd="./requirements.sh"
echo ">> $cmd"; $cmd
cd ..

# check if other dependent software are installed
echo
echo "Check if other dependent software are installed..."
cmd="python ${SCT_DIR}/scripts/sct_check_dependences.py"
echo ">> $cmd"; $cmd

# go to testing folder
path_toolbox_temp='pwd'
cd ../sct_testing/
path_testing='pwd' 

# display stuff
echo ""
echo "" "========================================================================================"
echo ""
echo "Done! If you had errors, please start a new Terminal and run the following command:"
echo "> sct_check_dependences.py"
echo
echo "If you are still getting errors, please post an issue here: https://sourceforge.net/p/spinalcordtoolbox/discussion/help/"
echo "or contact the developers."
echo
echo "You can now delete this folder by typing:"
echo "> cd .."
echo "> rm -rf ${path_toolbox_temp}*"
echo
echo "To get started, open a new Terminal and go to the testing folder:"
echo "> cd $path_testing"
echo "and follow instructions here: https://sourceforge.net/p/spinalcordtoolbox/wiki/get_started/"
echo
echo "To see all commands available, start a new Terminal and type \"sct\" then backspace"


