#/bin/sh
source $HOME/build/neuropoly/spinalcordtoolbox/python/bin/activate
export PYTHONPATH=$PYTHONPATH:$SCT_DIR/scripts
pytest -v --cov-append --cov=spinalcordtoolbox --cov=scripts unit_testing/*
