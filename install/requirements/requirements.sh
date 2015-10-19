#!/bin/bash
# 
#Installer for requirements
#
echo "--------------------------"
echo "INSTALLER FOR REQUIREMENTS"
echo "--------------------------"
# Installation using Conda
if which conda >/dev/null; then
    echo "* CONDA *"
    conda install --yes --file requirementsConda.txt
else
    echo "ERROR: Conda is not installed. Cannot install dependences."
fi
# Installation using Pip
if which pip >/dev/null; then
    echo "* PIP *"
    pip install -r requirementsPip.txt
else
    echo "ERROR: Pip is not installed. Cannot install dependences."
fi
