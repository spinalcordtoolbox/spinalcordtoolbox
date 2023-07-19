.. _normalizing-morphometrics:

Normalizing morphometrics to asses spinal cord compression
##########################################################


Compute ratio of **AP diameter** normalized with healthy controls:
--------------------------------------------------------------------------------
.. code:: sh

   sct_compute_compression -i t2_compressed_seg.nii.gz -vertfile t2_compressed_seg_labeled.nii.gz -l t2_compressed_compression_labels.nii.gz -normalize-hc 1 -o ap_ratio_norm_PAM50.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vertfile`` : Vertebral labeling file.
   - ``-l`` : Compression labels file.
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``ap_ratio_norm_PAM50.csv`` : A file containing the ratio values for each. This file is partially replicated in the table below.

.. csv-table:: Anterior and posterior diameter ratio with levels above and below all compressions.
   :file: ap_ratio_norm_PAM50.csv
   :header-rows: 1