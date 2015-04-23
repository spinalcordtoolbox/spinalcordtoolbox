#!/bin/bash
#
# Installer for spinal cord toolbox for developper.
#
# This script will update the bashrc and bash_profile for environment path setting.
# Must be run as a non-administrator (no sudo).
# Installation location: current path

# parameters
PATH_INSTALL=$(cd $(dirname "$0"); pwd)
PATH_INSTALL=$(dirname ${PATH_INSTALL})
PATH_INSTALL=$(dirname ${PATH_INSTALL})
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
-h                          display this help

EOF
}

echo
echo "============================="
echo "SPINAL CORD TOOLBOX INSTALLER"
echo "============================="

# check if user is sudoer
if [ "$(whoami)" == "root" ]; then
  echo "Sorry, you are root. Please type: ./installer_dev.sh without sudo. Your password will be required later."
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
  echo "ERROR: The path you entered does not exist: ${PATH_INSTALL}. Create it first."
  exit
fi

# check if last character is /. If not, add it.
LEN=${#PATH_INSTALL}-1
if [ "${PATH_INSTALL}" != "/" ]; then
  PATH_INSTALL=$PATH_INSTALL"/"
fi

# Set toolbox installation path
SCT_DIR="${PATH_INSTALL}spinalcordtoolbox"
echo "SCT_DIR=$SCT_DIR"

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
export PATH=${PATH}:$SCT_DIR/bin
if [ "$(uname -s)" == "Linux" ]; then
  echo 'export PATH=${PATH}:$SCT_DIR/bin/linux' >> ~/.bashrc
  export PATH=${PATH}:$SCT_DIR/bin/linux
elif [ "$(uname -s)" == "Darwin" ]; then
  echo 'export PATH=${PATH}:$SCT_DIR/bin/osx' >> ~/.bashrc
  export PATH=${PATH}:$SCT_DIR/bin/osx
fi
echo 'export PATH=${PATH}:$SCT_DIR/scripts' >> ~/.bashrc
export PATH=${PATH}:$SCT_DIR/scripts
# add PYTHONPATH variable to allow import of modules
echo 'export PYTHONPATH=${PYTHONPATH}:$SCT_DIR/scripts' >> ~/.bashrc
export PYTHONPATH=${PYTHONPATH}:$SCT_DIR/scripts
echo 'export SCT_DIR PATH' >> ~/.bashrc
export SCT_DIR PATH
# forbid to run several ITK instances in parallel (see issue #201).
NUMBEROFCORES=`sysctl hw.ncpu | awk '{print $2}'`
echo "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NUMBEROFCORES" >> ~/.bashrc
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NUMBEROFCORES
echo "PATH=${PATH}"

# check if .bash_profile exists. If so, we check if link to .bashrc is present in it. If not, we add it at the end.
if [ -e "~/.bash_profile" ]; then
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
source ~/.bashrc

# install required software
echo
echo " Install required software... "
cd requirements
cmd="./requirements.sh"
echo ">> $cmd"; $cmd
cd ..

# Create links to python scripts
echo
echo "Create links to python scripts..."
cmd="${SCT_DIR}/install/create_links.sh -a"
echo ">> $cmd"; $cmd

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
echo "To get started, open a new Terminal and follow instructions here: https://sourceforge.net/p/spinalcordtoolbox/wiki/get_started/"


