#/bin/sh
source $HOME/build/neuropoly/spinalcordtoolbox/python/bin/activate
export PATH=$PATH:$SCT_DIR/scripts
export PYTHONPATH=$PYTHONPATH:$SCT_DIR/scripts
echo $PATH
echo $PYTHONPATH
pytest -v --cov=spinalcordtoolbox.deepseg_gm unit_testing
