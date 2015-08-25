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
    echo "* PIP *"
    pip install -r requirementsPip.txt
else
    echo "Conda is not installed"
    echo "Cannot install requirements "
fi
