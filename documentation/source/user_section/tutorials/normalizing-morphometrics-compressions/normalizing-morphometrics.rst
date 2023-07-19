.. _normalizing-morphometrics:

Normalizing morphometric to asses spinal cord compression
#########################################################


Compute ratio of AP diameter normalized with a database of healthy controls:

.. code:: sh

   sct_compute_compression -i t2_compressed_seg.nii.gz -l t2_compressed_compression_labels.nii.gz -vertfile t2_compressed_seg_labeled.nii.gz -normalize-hc 1 -o ap_ratio_norm_PAM50.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vert`` : The vertebral levels to compute metrics across. Vertebral levels can be specified individually (``3,4``) or as a range (``3:4``).
   - ``-vertfile`` : The volume containing vertebral levels. Here, we use a PAM50 template object that had been previously warped to the same coordinate space as the input segmentation.
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``csa_c3c4.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.

.. .. csv-table:: CSA values computed for C3 and C4 vertebral levels (Averaged)
..    :file: csa_c3c4.csv
..    :header-rows: 1