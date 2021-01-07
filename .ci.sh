#!/usr/bin/env bash
# CI testing script
#  Installs SCT from scratch and runs all the tests we've ever written for it.

# stricter shell mode
# https://sipb.mit.edu/doc/safe-shell/
set -eo pipefail  # exit if non-zero error is encountered (even in a pipeline)
set -u            # exit if unset variables used
shopt -s failglob # error if a glob doesn't find any files, instead of remaining unexpanded

export SCT_PROGRESS_BAR=off # disable SCT's progress bar (e.g. for sct_download_data)
export PIP_PROGRESS_BAR=off # disable pip's progress bar for the duration of CI
export PY_COLORS=1 # Colored pytest output (https://github.com/pytest-dev/pytest/issues/7443#issuecomment-656642591)

install_sct () {
  # NB: we only force in-place (-i) installs to avoid pytest running from the source
  #     instead of the installed folder, where the extra detection models are.
  #     Further explanation at https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
  #     TO BE REMOVED during https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3140.
  # NB: '-c' disables sct_check_dependencies so we can check it
  # NB: '-k' keeps the folders 'data/', 'bin/', and 'python/' to speed up installation
  ./install_sct -iyck
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
  pytest testing/api testing/cli
  # NB: 'testing/batch_processing' is run by a separate CI workflow
}

run_tests_with_coverage(){
  # NB: Testing using example from https://github.com/codecov/example-python
  activate_venv_sct
  pytest --cov=spinalcordtoolbox --cov-config setup.cfg --cov-branch --cov-report=xml:cov-api.xml testing/api
  pytest --cov=spinalcordtoolbox --cov-config setup.cfg --cov-branch --cov-report=xml:cov-cli.xml testing/cli
  # NB: 'testing/batch_processing' can't easily be coverage by codecov, as
  # the actual processing is invoked via a shell script, rather than pytest
}

while getopts ":ictv" opt; do
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
  v)
    run_tests_with_coverage
    ;;
  *)
    exit 99
    ;;
  esac
done
