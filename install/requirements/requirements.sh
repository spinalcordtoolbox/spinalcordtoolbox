#!/bin/bash
# 
#Installer for following requirements:
#
#	using conda:
#		numpy
#		scipy
#		sympy
#		matplotlib
#		pip
#
#	using pip:
#		nibadel
echo "-------------------------------"
echo "INSTALLER FOR SOME REQUIREMENTS"
echo "-------------------------------"
echo "* CONDA *"
CURRENT_PATH=$PWD

conda install --yes --file requirementsConda.txt
echo "* PIP *"
pip install -r requirementsPip.txt
