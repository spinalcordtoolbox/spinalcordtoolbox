#!/usr/bin/env bash
# CI testing script
#  Installs SCT from scratch and runs all the tests we've ever written for it.

# stricter shell mode
# https://sipb.mit.edu/doc/safe-shell/
set -eo pipefail  # exit if non-zero error is encountered (even in a pipeline)
set -u            # exit if unset variables used
shopt -s failglob # error if a glob doesn't find any files, instead of remaining unexpanded

export PIP_PROGRESS_BAR=off # disable pip's progress bar for the duration of CI
export PY_COLORS=1 # Colored pytest output (https://github.com/pytest-dev/pytest/issues/7443#issuecomment-656642591)

install_sct () {
  # NB: we only force in-place (-i) installs to avoid pytest running from the source
  #     instead of the installed folder, where the extra detection models are.
  #     Further explanation at https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
  #     TO BE REMOVED during https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3140.
  # NB: '-c' disables sct_check_dependencies so we can check it separately
  ./install_sct -iyc
}

activate_venv_sct(){
  source python/etc/profile.d/conda.sh  # to be able to call conda
  set +u
  conda activate venv_sct
  set -u
}

check_dependencies() {
  activate_venv_sct
  sct_check_dependencies
}

run_tests() {
  activate_venv_sct
  pytest
}

run_tests_with_coverage(){
  # NB: Coverage does not currently work: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2702
  activate_venv_sct
  pip install coverage
  echo -ne "import coverage\ncov = coverage.process_startup()\n" > sitecustomize.py
  echo -ne "[run]\nconcurrency = multiprocessing\nparallel = True\n" > .coveragerc
  COVERAGE_PROCESS_START="$PWD/.coveragerc" COVERAGE_FILE="$PWD/.coverage" pytest
  coverage combine
}

while getopts ":ict" opt; do
  case $opt in
  i)
    install_sct
    ;;
  c)
    check_dependencies
    ;;
  t)
    run_tests
    ;;
  *)
    exit 99
    ;;
  esac
done
