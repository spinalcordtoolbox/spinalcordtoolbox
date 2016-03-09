#!/bin/bash
# USAGE
# > bash install_w_conda.sh
#
# This is the spinalcord toolbox (SCT) installer
# It downloads the Conda (http://conda.pydata.org/) version
# of python and install the SCT requirements over it
#
# The SCT can be install in place where you dowload it. It
# is the default installation in user mode. But then do no
# delete the source code or you will delete the installation too!
#
# If you run the installer as super user, the default install is /opt, if you
# choose this option or any other directory other than the source location,
# you can get rid of the source code after the installation is successful.
#
#


# TODO add some doc to the installer
# TODO: remove create_links, installer_dev.sh, create_launchs, install_ants_osx.sh, install_external.py
# TODO: rename this installer to "install_sct"


SCT_FOLDER_NAME="spinalcordtoolbox"

echo "Welcome to the SCT installer!"

START_DIR=$PWD
function finish {
#  Clean at exit
  cd ${START_DIR}
}
trap finish EXIT

# Check where the install script is ran from
if [[ $PWD =~ /spinalcordtoolbox$ ]] ;then
  SCT_SOURCE=${PWD}
elif  [[ $PWD =~ /spinalcordtoolbox/install$ ]] ;then
  SCT_SOURCE=${PWD%/install}
elif  [ ls $SCT_FOLDER_NAME ] ;then
  SCT_SOURCE=${PWD}/${SCT_FOLDER_NAME}
else
   echo I can\'t find SCT source folder \"${SCT_FOLDER_NAME}/\"
fi

# TODO: deal with OSX which does not have .bashrc (and not sourced in bash_profile)
if [[ $UID == 0 ]]; then
  # sudo mode
  THE_BASHRC=/etc/profile.d/sct.sh
  THE_CSHRC=/etc/profile.d/sct.csh
  INSTALL_DIR=/opt
else
  # user mode
  THE_BASHRC=${HOME}/.bashrc
  THE_CSHRC=${HOME}/.cshrc
  INSTALL_DIR=${SCT_SOURCE%/*}
fi
# !!! DEBUG
#INSTALL_DIR=/home/poquirion/test

# Set install dir
while  true ; do
  echo SCT will be installed here: [${INSTALL_DIR}]
  echo -n Type Enter or a different path:
  read new_install
  if [ -d "${new_install}" ]; then
    INSTALL_DIR=${new_install}
    break
  elif [ ! "${new_install}" ]; then
    break
  else
    echo invalid directory
    continue
  fi
done

SCT_DIR=${INSTALL_DIR%/}/${SCT_FOLDER_NAME}

# create fetching variables for Python packages
if uname -a | grep -i  darwin > /dev/null 2>&1; then
    # Do something under Mac OS X platformn
  OS=osx
  conda_installer=$(find ${SCT_SOURCE}/external -type f -name "Miniconda*OSX*")
  dipy_whl=$(find ${SCT_SOURCE}/external -type f -name "dipy*osx*")
  ornlm_whl=$(find ${SCT_SOURCE}/external -type f -name "ornlm*osx*")
  echo OSX MACHINE
elif uname -a | grep -i  linux > /dev/null 2>&1; then
  OS=linux
  conda_installer=$(find ${SCT_SOURCE}/external -type f -name "Miniconda*Linux*")
  dipy_whl=$(find ${SCT_SOURCE}/external -type f -name "dipy*linux*")
  ornlm_whl=$(find ${SCT_SOURCE}/external -type f -name "ornlm*linux*")
  echo LINUX MACHINE
else
  echo Sorry, the installer only supports Linux and OSX, quitting installer
  exit 1
fi

# Copy files to destination directory
if [ "${SCT_DIR}" != "${SCT_SOURCE}" ]; then
  echo copying source files to "${SCT_DIR}"
  mkdir -p ${SCT_DIR}/bin
  cd ${SCT_SOURCE}/bin
  find  . -type f -not -path '*/miniconda/*' -exec cp -v --parent {} ${SCT_DIR}/bin \;
  cd ${START_DIR}
  cp -r ${SCT_SOURCE}/scripts  ${SCT_DIR}/.
  cp ${SCT_SOURCE}/version.txt  ${SCT_DIR}/.
  mkdir -p ${SCT_DIR}/install
  cp -rp ${SCT_SOURCE}/install/requirements ${SCT_DIR}/install
  cp -rp ${SCT_SOURCE}/testing ${SCT_DIR}/.
fi

echo installing conda ...

bash ${conda_installer} -p ${SCT_DIR}/bin/${OS}/miniconda -b -f

. ${SCT_DIR}/bin/${OS}/miniconda/bin/activate ${SCT_DIR}/bin/${OS}/miniconda
echo ${SCT_DIR}/bin/${OS}/miniconda/bin/activate

echo Installing dependencies
# N.B. the flag --ignore-installed is required because if user already has other dependencies, it will not install
conda install --yes --file ${SCT_SOURCE}/install/requirements/requirementsConda.txt
pip install --ignore-installed  -r ${SCT_SOURCE}/install/requirements/requirementsPip.txt
pip install --ignore-installed  ${dipy_whl}  ${ornlm_whl}

echo All requirement installed

# update PATH environment
# TODO: do not add path if sudo, but display message
# TODO: change intro line for: SPINALCORDTOOLBOX (installed on 2016-03-09 10:31)
# TODO: check if line is there in case of re-installation
# TODO: deal with OSX which does not source .bashrc: add to bash_profile (create if does not exist)
while  [[ ! ${add_to_path} =~ ^([Yy](es)?|[Nn]o?)$ ]] ; do
  echo -n "Do you want to add the sct_* script to your PATH environment? Yes/No: "
  read add_to_path
done
echo ""
if [[ ${add_to_path} =~ ^[Yy] ]]; then
  # assuming bash
  echo "# SPINALCORDTOOLBOX" > ${THE_BASHRC}
  echo "export PATH=${SCT_DIR}/bin:\$PATH" >> ${THE_BASHRC}
  # (t)csh for good measure
  echo "# SPINALCORDTOOLBOX" >> ${THE_CSHRC}
  echo "setenv PATH \"${SCT_DIR}/bin:\$PATH\"" ${THE_CSHRC}
else
   echo Not adding ${INSTALL_DIR} to \$PATH
   echo You can always add it later or call SCT functions with full path ${SCT_DIR}/bin/sct_function
fi

#Make sure sct script are executable
find ${SCT_DIR}/bin/ -maxdepth 2 -type f -exec chmod 755 {} \;

# Run check dependencies
# TODO: update to check_dependencies
${SCT_DIR}/bin/sct_check_dependences
echo INSTALLATION DONE
