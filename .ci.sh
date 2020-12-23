#!/bin/bash
# CI testing script
#  Installs SCT from scratch and runs all the tests we've ever written for it.

set -e # Error build immediately if install script exits with non-zero

echo Installing SCT
PIP_PROGRESS_BAR=off ./install_sct -y
echo $?
echo "... STATUS"

echo *** CHECK PATH ***
ls -lA bin  # Make sure all binaries and aliases are there
source python/etc/profile.d/conda.sh  # to be able to call conda
conda activate venv_sct  # reactivate conda for the pip install below

echo *** UNIT TESTS ***
pip install coverage
echo -ne "import coverage\ncov = coverage.process_startup()\n" > sitecustomize.py
echo -ne "[run]\nconcurrency = multiprocessing\nparallel = True\n" > .coveragerc
COVERAGE_PROCESS_START="$PWD/.coveragerc" COVERAGE_FILE="$PWD/.coverage" \
  pytest
coverage combine

# TODO: move this part to a separate travis job; there's no need for each platform to lint the code
echo *** ANALYZE CODE ***
pip install pylint
bash -c 'PYTHONPATH="$PWD/scripts:$PWD" pylint -j3 --py3k --output-format=parseable --errors-only $(git ls-tree --name-only -r HEAD | sort | grep -E "(spinalcordtoolbox|scripts|testing).*\.py" | xargs); exit $(((($?&3))!=0))'

# python create_package.py -s ${TRAVIS_OS_NAME}  # test package creation
# cd ../spinalcordtoolbox_v*
# yes | ./install_sct  # test installation of package

