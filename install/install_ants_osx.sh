#!/bin/bash
#
# Installer for ANTs. Must be run as an administrator.
# Installation location: /usr/local/ants/
#
# Author: Julien Cohen-Adad <jcohen@polymtl.ca>
# Modified: 2014-05-27


# parameters
#PATH_INSTALL="/usr/local/"
URL_ANTS="https://dl.dropboxusercontent.com/u/20592661/spinalcordtoolbox/ants.tar.gz"
PATH_ANTS="/usr/local/ants"
PATH_INSTALL="/usr/local/"
echo


# check if user is sudoer
if [ "$(whoami)" != "root" ]; then
  echo "Sorry, you are not root. Please type: sudo ./install_ants.sh"
  echo
  exit
fi

echo ============
echo INSTALL ANTS
echo ============
# check if folder already exists - if so, delete it
echo
echo "Check if ants folder is already there (if so, delete it)..."
if [ -e "${PATH_ANTS}" ]; then
  echo ".. Yup! found it. "
  # check if user is sure to delete it
  echo
  while true; do
      read -p 'Are you sure you want to remove your current version of ANTS? [yes/no]: ' yn
      case $yn in
          [Yy]* ) break;;
          [Nn]* ) echo "Exit program"; exit;;
          * ) echo "Please answer yes or no.";;
      esac
  done
  cmd="rm -rf ${PATH_ANTS}"
  echo ">> $cmd"; $cmd
else
  echo ".. Nope! Doesn't seem to be there"
fi

# download file from internet
echo
echo "Download file from internet..."
cmd="curl -O ${URL_ANTS}"
echo ">> $cmd"; $cmd

# decompress file
echo
echo "Decompress file..."
cmd="tar -zxvf ants.tar.gz"
echo ">> $cmd"; $cmd

# move folder
echo
echo "Move folder to: "
cmd="mv ants ${PATH_INSTALL}"
echo ">> $cmd"; $cmd

# edit .bash_profile
echo
echo "Edit .bash_profile..."
if grep -q "ANTS" ~/.bash_profile; then
  echo ".. It seems like .bash_profile already includes ANTS path."
else
  echo '' >> ~/.bash_profile
  echo '# ANTS' >> ~/.bash_profile
  echo 'PATH=${PATH}:/usr/local/ants/bin' >> ~/.bash_profile
fi

# display stuff
echo
echo "---"
echo "Done! Please restart the Terminal to load environment variables."
echo "---"
echo
