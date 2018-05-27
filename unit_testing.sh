#/bin/sh
source $HOME/build/neuropoly/spinalcordtoolbox/python/bin/activate
export PYTHONPATH=$PYTHONPATH:$SCT_DIR/scripts
pytest -v --cov=spinalcordtoolbox.deepseg_gm unit_testing/test_deepseg_gm.py
pytest -v --cov=spinalcordtoolbox.metadata unit_testing/test_metadata.py
