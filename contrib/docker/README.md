# Minimal SCT Docker file w/ Ubuntu 16.04

This directory contains a minimal Dockerfile to install spinalcoordtoolbox.

## Instructions

Inside this directory, run:

    docker build -t dockersct .
    docker run --name sctcontainer -i -t dockersct
    
# SCT installation with GUI

This directory also contains launcher scripts for XMing. These are used in the "Install SCT using Docker" steps found in the ReadTheDocs documentation.
