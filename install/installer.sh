#!/bin/bash
#
# Installer for spinal cord toolbox.
#
# This script will install the spinal cord toolbox under and configure your environment.
# Must be run as a non-administrator (no sudo).
# Installation location: /usr/local/spinalcordtoolbox/

# parameters
PATH_INSTALL="/usr/local/"
ISSUDO="sudo "

function usage()
{
cat << EOF

`basename ${0}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
Install the Spinal Cord Toolbox

USAGE
`basename ${0}` -p <path>

MANDATORY ARGUMENTS
-p <path>                   installation path
-h                          display this help

EOF
}

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

while getopts “hp:” OPTION
do
    case $OPTION in
        h)
            usage
            exit 1
            ;;
        p)
            PATH_INSTALL=$OPTARG
            ISSUDO=""
            ;;
        ?)
            usage
            exit
            ;;
    esac
done

if [[ ! -d $PATH_INSTALL ]]; then
  echo "ERROR: Installation path is not correct: ${PATH_INSTALL}. Exit program."
  exit
fi

# check if last character is /. If not, add it.
LEN=${#PATH_INSTALL}-1
if [ "${PATH_INSTALL}" != "/" ]; then
  PATH_INSTALL=$PATH_INSTALL"/"
fi

# Set toolbox installation path
SCT_DIR="${PATH_INSTALL}spinalcordtoolbox"


# check if folder already exists - if so, delete it
echo
echo "Check if spinalcordtoolbox is already installed (if so, delete it)..."
if [ -e "${SCT_DIR}" ]; then
  cmd="${ISSUDO}rm -rf ${SCT_DIR}"
  echo ">> $cmd"; $cmd
fi

# create folder
echo
echo "Create folder: ${PATH_INSTALL}spinalcordtoolbox..."
cmd="${ISSUDO}mkdir ${SCT_DIR}"
echo ">> $cmd"; $cmd

# copy files
echo
echo "Copy toolbox..."
cmd="${ISSUDO}cp -r spinalcordtoolbox/* ${SCT_DIR}"
echo ">> $cmd"; $cmd

# check if .bashrc was already modified. If so, we delete lines related to SCT
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
echo 'export PATH=${PATH}:$SCT_DIR/bin' >> ~/.bashrc
# add PYTHONPATH variable to allow import of modules
echo 'export PYTHONPATH=${PYTHONPATH}:$SCT_DIR/scripts' >> ~/.bashrc
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

# get current path
path_toolbox_temp=`pwd`

# display stuff
echo ""
echo "" "========================================================================================"
echo ""
echo "Done! If you had errors, please start a new Terminal and run the following command:"
echo "> sct_check_dependences"
echo
echo "If you are still getting errors, please post an issue here: https://sourceforge.net/p/spinalcordtoolbox/discussion/help/"
echo "or contact the developers."
echo
echo "You can now delete this folder by typing:"
echo "> cd .."
echo "> rm -rf ${path_toolbox_temp}"
echo
echo "To get started, open a new Terminal and follow instructions here: https://sourceforge.net/p/spinalcordtoolbox/wiki/get_started/"


