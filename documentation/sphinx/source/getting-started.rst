Getting Started
###############


To see all the commands available from SCT, start a new Terminal and type `sct` then press "tab".


SCT includes a `batch_processing <https://github.com/neuropoly/spinalcordtoolbox/blob/master/batch_processing.sh>`_
script that will get you started with basic analyses.

.. code:: sh

   $ ~/sct_*/batch_processing.sh

The script source is reasonably documented.

This script uses the T2-weighted image as the anatomical image for registration to the template. Once you've run and understood the script, you can try to modify it using the T1-weighted MPRAGE instead of the T2-weighted.


