#!/usr/bin/env bash
# CI testing script
#  Installs SCT from scratch and runs all the tests we've ever written for it.

# stricter shell mode
# https://sipb.mit.edu/doc/safe-shell/
set -eo pipefail  # exit if non-zero error is encountered (even in a pipeline)
set -u            # exit if unset variables used
shopt -s failglob # error if a glob doesn't find any files, instead of remaining unexpanded

export PIP_PROGRESS_BAR=off # disable pip's progress bar for the duration of CI

echo Installing SCT
# NB: we only force in-place (-i) installs to avoid pytest running from the source
#     instead of the installed folder, where the extra detection models are.
#     Further explanation at https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
#     TO BE REMOVED during https://github.com/neuropoly/spinalcordtoolbox/issues/3140.
./install_sct -iy

source python/etc/profile.d/conda.sh  # to be able to call conda
conda activate venv_sct  # reactivate conda for the pip install below

echo *** UNIT TESTS ***
pip install coverage
echo -ne "import coverage\ncov = coverage.process_startup()\n" > sitecustomize.py
echo -ne "[run]\nconcurrency = multiprocessing\nparallel = True\n" > .coveragerc
COVERAGE_PROCESS_START="$PWD/.coveragerc" COVERAGE_FILE="$PWD/.coverage" \
  pytest
coverage combine
