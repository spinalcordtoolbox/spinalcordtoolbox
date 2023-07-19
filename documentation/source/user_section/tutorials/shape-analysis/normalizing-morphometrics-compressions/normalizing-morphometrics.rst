.. _normalizing-morphometrics:

Normalizing morphometrics
#############################

Compute ratio of **AP diameter** normalized with healthy controls:
--------------------------------------------------------------------------------
First, we will start by computing the ratio of the anterior-posterior diameter at the level of compression and levels above and below all compresssions.
We will also use the flag ``-normalize-hc`` to use a database of healthy controls to normalize the anterior-posterior diameters.`
.. code:: sh

   sct_compute_compression -i t2_compressed_seg.nii.gz -vertfile t2_compressed_seg_labeled.nii.gz -l t2_compressed_compression_labels.nii.gz -metric diameter_AP -normalize-hc 1 -o ap_ratio_norm_PAM50.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vertfile`` : Vertebral labeling file.
   - ``-l`` : Compression labels file.
   - ``metric``: Metric to compute ratio: diameter_AP (default).
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``ap_ratio_norm_PAM50.csv`` : A file containing the ratio values for each. This file is partially replicated in the table below.
       - **diameter_AP_ratio**: Ratio computed in the subject's native space.
       - **diameter_AP_ratio_PAM50**: Ratio computed in the PAM50 space.
       - **diameter_AP_ratio_PAM50_normalized**: Ratio computed in the PAM50 space and nromalized with healthy controls

.. csv-table:: Anterior - posterior diameter ratio with levels above and below all compressions.
   :file: ap_ratio_norm_PAM50.csv
   :header-rows: 1


Compute ratio of **Area (CSA)** normalized with healthy controls:
--------------------------------------------------------------------------------
.. code:: sh

   sct_compute_compression -i t2_compressed_seg.nii.gz -vertfile t2_compressed_seg_labeled.nii.gz -l t2_compressed_compression_labels.nii.gz -metric area -normalize-hc 1 -o area_ratio_norm_PAM50.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vertfile`` : Vertebral labeling file.
   - ``-l`` : Compression labels file.
   - ``metric``: Metric to compute ratio: area.
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``area_ratio_norm_PAM50.csv`` : A file containing the ratio values for each. This file is partially replicated in the table below.

.. csv-table:: Cross-sectional area ratio with levels above and below all compressions.
   :file: area_ratio_norm_PAM50.csv
   :header-rows: 1