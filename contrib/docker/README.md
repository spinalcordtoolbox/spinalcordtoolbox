# Minimal SCT Docker file w/ Ubuntu 22.04

This directory contains a minimal Dockerfile to install spinalcoordtoolbox.

## Instructions

Inside this directory, run:

    docker build -t dockersct .
    docker run --name sctcontainer -it dockersct

For more a more detailed set of instructions, refer to the documentation at <https://spinalcordtoolbox.com/latest/user_section/installation/docker.html>
