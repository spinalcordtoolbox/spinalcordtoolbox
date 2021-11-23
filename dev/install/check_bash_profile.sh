#!/bin/bash
#
# Check .bash_profile for all dependences and modify it if necessary.
#

# And here we have Bash Patterns:
echo -n "Checking if c3d is installed...                   " 
C3D=`find / -name "c3d" -type f -print -quit 2>/dev/null`
if [[ "$C3D" == *c3d ]]
then
    echo '[OK]'
else
    echo '[WARNING]: c3d is NOT installed. Please install it from there: http://www.itksnap.org/'
	exit
fi

echo -n "Checking if c3d is declared in .bash_profile...   "
PATHC3D=`which c3d`
if [[ $PATHC3D != '' ]]; then
  echo '[OK]'
else
  echo '[WARNING]: c3d not declared. Now modifying ~/.bash_profile...'
  echo "# C3D (added on $(date +%Y-%m-%d))" >> ~/.bash_profile
  echo "PATH=${PATH}:${C3D%????}" >> ~/.bash_profile
fi

echo "Done! Please restart your Terminal for changes to take effect."
echo
