CSA (Averaged across vertebral levels)
######################################

First, we will start by computing the cord cross-sectional area (CSA) averaged across vertebral levels. As an example, we'll choose the C3 and C4 vertebral levels, but you can specify any vertebral levels present in the vertebral level file.

.. code:: sh

   sct_process_segmentation -i t2_seg.nii.gz -vert 3:4 -vertfile ./label/template/PAM50_levels.nii.gz -o csa_c3c4.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vert`` : The vertebral levels to compute metrics across. Vertebral levels can be specified individually (``3,4``) or as a range (``3:4``).
   - ``-vertfile`` : The volume containing vertebral levels. Here, we use a PAM50 template object that had been previously warped to the same coordinate space as the input segmentation.
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``csa_c3c4.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.

.. csv-table:: CSA values computed for C3 and C4 vertebral levels (Averaged)
   :file: csa_c3c4.csv
   :widths: 14, 9, 13, 8, 7, 7, 7, 8, 7, 8, 7
   :header-rows: 1