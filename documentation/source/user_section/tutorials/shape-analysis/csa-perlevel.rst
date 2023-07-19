.. _csa-perlevel:

CSA (Per level)
###############

Next, we will compute CSA for each individual vertebral level (rather than averaging).

.. code:: sh

   sct_process_segmentation -i t2_seg.nii.gz -vert 3:4 -vertfile ./label/template/PAM50_levels.nii.gz -perlevel 1 -o csa_perlevel.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vert`` : The vertebral levels to compute metrics across. Vertebral levels can be specified individually (``3,4``) or as a range (``3:4``).
   - ``-vertfile`` : The label file that specifies vertebral levels. Here, we use the PAM50 template object that had been previosuly warped to the same coordinate space as the T2 segmentation.
   - ``-perlevel`` : Set this option to 1 to turn on per-level computation.
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``csa_perlevel.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.

.. csv-table:: CSA values computed for C3 and C4 vertebral levels
   :file: csa_perlevel.csv
   :header-rows: 1