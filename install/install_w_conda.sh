#!/bin/bash
#USAGE
#  ./install_w_conda.sh


SCT_SOURCE=${PWD}
SCT_SOURCE=${SCT_SOURCE%/}
SCT_SOURCE=${SCT_SOURCE%/*}

INSTALL_DIR=${SCT_SOURCE}

SUDO=
if [[ $UID == 0 ]]; then SUDO=1 ;fi

echo "Welcome to the SCT installer V2.0 "

if uname -a | grep -i  darwin > /dev/null 2>&1; then
    # Do something under Mac OS X platformn
  OS=osx
  conda_installer=Miniconda-latest-MacOSX-x86_64.sh
  echo OSX MACHINE
elif uname -a | grep -i  linux > /dev/null 2>&1; then
  OS=linux
  conda_installer=Miniconda-latest-Linux-x86_64.sh
  echo LINUX MACHINE
else
  echo Sorry, the installer only support Linux and OSX
  exit 1
fi

echo installing conda ...

bash ${SCT_SOURCE}/external/${conda_installer} -p ${SCT_SOURCE}/bin/${OS}/miniconda -b -f

. ${SCT_SOURCE}/bin/${OS}/miniconda/bin/activate ${SCT_SOURCE}/bin/${OS}/miniconda
echo ${SCT_SOURCE}/bin/${OS}/miniconda/bin/activate

echo Installing dependencies
conda install --yes --file ${SCT_SOURCE}/install/requirements/requirementsConda.txt
pip install -r ${SCT_SOURCE}/install/requirements/requirementsPip.txt


read -p "Do you want to be add the sct_* script to your paths? Y/N " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
  # assuming bash
  echo "#SPINALCORDTOOLBOX PATH" #>> ${HOME}/.bashrc
  echo "export PATH=${INSTALL_DIR}/bin:\$PATH" #>> ${HOME}/.bashrc
  if [ -e ${HOME}/.cshrc ]; then
    # (t)csh for good mesure
    echo "#SPINALCORDTOOLBOX PATH" #>> ${HOME}/.cshrc
    echo "setenv PATH ${INSTALL_DIR}/bin:\$PATH" #>> ${HOME}/.cshrc
  fi
else
   echo Not adding ${INSTALL_DIR} to \$PATH
   echo you can always add it later or call sct_FUNCTIONS directly from ${INSTALL_DIR}/bin
fi

# TODO edit sct_fct to get a default fallback to the local sct_laucher
. ${INSTALL_DIR}/bin/sct_launcher
${INSTALL_DIR}/bin/sct_check_dependences
echo ISTALLATION SUCSESSFULL
