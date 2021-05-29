CSA (Per-slice)
###############

Finally, to compute CSA for individual slices, set the ``-perslice`` argument to 1, combined with the ``-z`` argument to specify slice numbers (or a range of slices).

.. code:: sh

   sct_process_segmentation -i t2_seg.nii.gz -perslice 1 -z 30:35 -o csa_perslice.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-perslice`` : Set this option to 1 to turn on per-slice computation.
   - ``-z`` : The Z-axis slices to compute metrics for. Slices can be specified individually (``30,31,32,33,34,35``) or as a range (``30:35``).
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``csa_perslice.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.


.. csv-table:: CSA values across slices 30 to 35
   :file: csa_perslice.csv
   :widths: 13, 9, 12, 8, 7, 7, 7, 9, 7, 9, 7
   :header-rows: 1