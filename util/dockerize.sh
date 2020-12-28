#!/usr/bin/env bash
# dockerize.sh: run a script inside of another.
#
# usage: DOCKER_IMAGE="<image>" DOCKER_DEPS_CMD="<command to run before script>" dockerize.sh script.sh

# stricter shell mode
# https://sipb.mit.edu/doc/safe-shell/
set -eo pipefail  # exit if non-zero error is encountered (even in a pipeline)
set -u            # exit if unset variables used
shopt -s failglob # error if a glob doesn't find any files, instead of remaining unexpanded

CONTAINER=$(docker run \
    --init \
    -it -d \
    --rm \
    -v "`pwd`":/repo -w /repo \
    "$DOCKER_IMAGE")
trap "docker stop "$CONTAINER"" EXIT
# set up a user:group matching that of the volume mount /repo, so the installer isn't confused
#
# TODO: it would be nice if the volume was mounted at `pwd`/, to further reduce the distinction
# between docker/nondocker runs, but docker gets the permissions wrong:
# it does `mkdir -p $mountpoint` *as root* so while the contents of the mountpoint are owned by
# $USER, its parents are owned by root, usually including /home/$USER which breaks things like pip.
# and there's no way to boot a container, `mkdir -p` manually, then attach the volume *after*.
docker exec "$CONTAINER" groupadd -g "`id -g`" "`id -g -n`"
docker exec "$CONTAINER" useradd -m -u "`id -u`" -g "`id -g`" "`id -u -n`"
# install platform-specific dependencies
if [ -n "$DOCKER_DEPS_CMD" ]; then
    docker exec "$CONTAINER" sh -c "$DOCKER_DEPS_CMD"
fi
# recurse to run the real test script
# --user `id -u` makes sure the build script is the owner of the files at /repo
# TODO: pass through the Travis envs: https://docs.travis-ci.com/user/environment-variables/
docker exec --user "`id -u`":"`id -g`" "$CONTAINER" "$1"
