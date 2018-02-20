# Minimal SCT Docker file w/ Ubuntu 16.04

This directory contains a minimal Dockerfile to install spinalcoordtoolbox.

## Instructions

Inside this directory, run:

    docker build -t dockersct .
    docker run --name sctcontainer -i -t dockersct
