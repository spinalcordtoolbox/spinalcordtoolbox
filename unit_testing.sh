#/bin/sh
source $HOME/build/neuropoly/spinalcordtoolbox/python/bin/activate
export PATH=$PATH:$SCT_DIR/scripts
pytest -v --cov=spinalcordtoolbox.deepseg_gm unit_testing
