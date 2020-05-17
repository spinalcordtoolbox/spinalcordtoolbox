#!/bin/bash
# TravisCI testing harness.
#  Supports running locally (i.e. on whatever platform Travis has loaded us in)
#  or in a docker container specified by $DOCKER_IMAGE.
#
# usage: .travis.sh
#
# e.g. DOCKER_IMAGE="centos:8" .travis.sh

set -e # Error build immediately if install script exits with non-zero

# if this is a docker job, run in the container instead; but if not just run it here.
if [ -n "$DOCKER_IMAGE" ]; then
    ./util/dockerize.sh ./.ci.sh
elif [ "${TRAVIS_OS_NAME}" = "windows" ]; then
    choco install wsl-ubuntu-1804
    brun="/c/ProgramData/chocolatey/lib/wsl-ubuntu-1804/tools/unzipped/ubuntu1804.exe run"
    $brun sudo apt update
    $brun sudo apt install -y gcc
    $brun ./.ci.sh
else
    ./.ci.sh
fi
