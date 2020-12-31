#!/usr/bin/env bash
# TravisCI testing harness.
#  Supports running locally (i.e. on whatever platform Travis has loaded us in)
#  or in a docker container specified by $DOCKER_IMAGE.
#
# usage: .travis.sh
#
# e.g. DOCKER_IMAGE="centos:8" .travis.sh

# stricter shell mode
# https://sipb.mit.edu/doc/safe-shell/
set -eo pipefail  # exit if non-zero error is encountered (even in a pipeline)
set -u            # exit if unset variables used
shopt -s failglob # error if a glob doesn't find any files, instead of remaining unexpanded

# if this is a docker job, run in the container instead; but if not just run it here.
if [ -n "${DOCKER_IMAGE:-}" ]; then
    if [ "$TRAVIS_OS_NAME" != "linux" ]; then
        echo "docker can only be used on linux" >&2
        exit 1
    fi
    ./util/dockerize.sh ./.ci.sh
elif [ -n "${WSL_IMAGE:-}" ]; then
    if [ "$TRAVIS_OS_NAME" != "windows" ]; then
        echo "WSL can only be used on windows" >&2
        exit 1
    fi
    choco install "$WSL_IMAGE" -y --ignore-checksums
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

    wsl -- "$WSL_DEPS_COMMAND"
    wsl -- ./.ci.sh
else
    ./.ci.sh
fi
