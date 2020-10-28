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
    travis_fold start "install.wsl-ubuntu-1804"
	      travis_time_start
            choco install wsl-ubuntu-1804 -y --ignore-checksums
        travis_time_finish
    travis_fold end "install.wsl-ubuntu-1804"

     # or, instead of choco, use curl + powershell:
     # https://docs.microsoft.com/en-us/windows/wsl/install-manual#downloading-distros-via-the-command-line
    # wsl --setdefault "Ubuntu-18.04"
     # TODO: Travis's version of wsl is too old for --setdefault.
     # Instead we trust that wsl will default to the installed
     # Ubuntu because it is the only option, but it would be
     # better to be explicit when it becomes possible.

    # disable apt's helpful (and build-breaking) interactive mode
    # https://linuxhint.com/debian_frontend_noninteractive/
    export DEBIAN_FRONTEND="noninteractive"
    # Use WSLENV to actually pass it into WSL
    # https://devblogs.microsoft.com/commandline/share-environment-vars-between-wsl-and-windows/
    export WSLENV=DEBIAN_FRONTEND

    travis_fold start "install.ubuntu-updates"
	      travis_time_start
            wsl apt-get update
        travis_time_finish
    travis_fold end "install.ubuntu-updates"

    travis_fold start "install.ubuntu-dependencies"
	      travis_time_start
            wsl apt-get install -y gcc git curl
            #wsl apt-get -y upgrade  # this step is probably important, but it's also sooo slow
        travis_time_finish
    travis_fold end "install.ubuntu-dependencies"
    wsl ./.ci.sh
else
    ./.ci.sh
fi
