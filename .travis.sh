#!/bin/bash
# CI testing script
# This is meant to be called from .travis.yml -- it works around limitations

set -e # Error build immediately if install script exits with non-zero

# if this is a docker job, set up and then recurse into the container,
# instead of continuing.
# TODO: figure out and pass needed environment variables individually with -e

if [ -n "$DOCKER_IMAGE" ]; then
   docker run \
     --name container \
     --init \
     -it -d \
     --rm \
     -v `pwd`:/repo -w /repo \
     "$DOCKER_IMAGE"
   trap "docker stop container" EXIT
   # set up a user:group matching that of the volume mount /repo, so the installer isn't confused
   docker exec container groupadd -g "`id -g`" `id -g -n`
   docker exec container useradd -m -u "`id -u`" -g "`id -g`" `id -u -n`
   if [ -n "$DOCKER_DEPS_CMD" ]; then
       docker exec container $DOCKER_DEPS_CMD    # install platform-specific dependencies
   fi
   # recurse to run the real test script
   # --user `id -u` makes sure the build script is the owner of the files at /repo
   # TODO: pass through the Travis envs: https://docs.travis-ci.com/user/environment-variables/
   docker exec --user `id -u`:`id -g` container "$0"
   exit 0
fi


echo Installing SCT
yes | ASK_REPORT_QUESTION=false ./install_sct
echo $?
echo "... STATUS"

echo *** CHECK PATH ***
ls -lA bin  # Make sure all binaries and aliases are there
source python/etc/profile.d/conda.sh  # to be able to call conda
conda activate venv_sct  # reactivate conda for the pip install below

echo *** UNIT TESTS ***
sct_download_data -d sct_testing_data  # for tests
pytest

echo *** INTEGRATION TESTS ***
pip install coverage
echo -ne "import coverage\ncov = coverage.process_startup()\n" > sitecustomize.py
echo -ne "[run]\nconcurrency = multiprocessing\nparallel = True\n" > .coveragerc
COVERAGE_PROCESS_START="$PWD/.coveragerc" COVERAGE_FILE="$PWD/.coverage" \
  sct_testing --abort-on-failure
coverage combine

# TODO: move this part to a separate travis job; there's no need for each platform to lint the code
echo *** ANALYZE CODE ***
pip install pylint
bash -c 'PYTHONPATH="$PWD/scripts:$PWD" pylint -j3 --py3k --output-format=parseable --errors-only $(git ls-tree --name-only -r HEAD | sort | grep -E "(spinalcordtoolbox|scripts|testing).*\.py" | xargs); exit $(((($?&3))!=0))'

#
# echo *** BUILD DOCUMENTATION ***
# pip install sphinx sphinxcontrib.programoutput sphinx_rtd_theme
# cd documentation/sphinx
# make html
# cd -

# python create_package.py -s ${TRAVIS_OS_NAME}  # test package creation
# cd ../spinalcordtoolbox_v*
# yes | ./install_sct  # test installation of package

