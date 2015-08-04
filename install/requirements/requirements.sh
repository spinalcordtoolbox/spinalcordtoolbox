#!/bin/bash
# 
#Installer for following requirements:
#
#	using conda:
#		numpy
#		scipy
#		sympy
#		matplotlib
echo "-------------------------------"
echo "INSTALLER FOR SOME REQUIREMENTS"
echo "-------------------------------"
if which conda >/dev/null; then
    echo "* CONDA *"
    conda install --yes --file requirementsConda.txt
    easy_install nibabel
    # pip install dipy
else
    echo "Conda is not installed"
    echo "Cannot install requirements "
fi
