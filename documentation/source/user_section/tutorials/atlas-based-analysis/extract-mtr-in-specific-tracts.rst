Using the atlas to extract MTR from specific white matter tracts
################################################################

Corticospinal tracts (averaged across 16 slices)
------------------------------------------------

Next, we will extract the MTR from the right and left corticospinal tracts.

.. code::

   sct_extract_metric -i mtr.nii.gz -f label/atlas -method map -l 4,5 -z 5:15 -o mtr_in_cst.csv

:Input arguments:
   - ``-i`` : Image to extract values from.
   - ``-f`` : File or folder used to pick out specific regions from the input image. In this case, we supply the folder of the white matter atlas that was transformed to the space of the MTR data.
   - ``-method`` : We specify ``map`` to choose the Maximum a Posteriori method, which helps to account for the partial volume effect. This method is most suitable for small regions. This method should only be used with the PAM50 white/gray matter atlas (or with any custom atlas) as long as the sum across all overlapping labels equals 1 in each voxel part of the atlas.
   - ``-l`` : The IDs of the label (or group of labels) to compute metrics for. In this case, labels 4 and 5 correspond to the left and right lateral corticospinal tracts. You can see the full list of labels to choose from by running ``sct_extract_metric -list-labels``.
   - ``-z`` : This argument further restricts the metric computation to specific slices. In this case, 16 slices in total (5:15) will be used for the averaging process.
   - ``-o`` : The name of the output file.

.. csv-table:: ``mtr_in_cst.csv``: MTR values in corticospinal tracts
   :file: mtr_in_cst.csv
   :header-rows: 1

We can see from the results that the MTR for both the left and right tracts is comparable, which is to be expected, given that the example data in this case was taken from a healthy adult.

Dorsal columns (averages across C2-C4 vertebral levels)
-------------------------------------------------------

Finally, we extract the MTR from the dorsal columns.

.. code::

   sct_extract_metric -i mtr.nii.gz -f label/atlas -method map -l 53 \
                      -vert 2:4 -vertfile label/template/PAM50_levels.nii.gz \
                      -o mtr_in_dc.csv

:Input arguments:
   - ``-i`` : Image to extract values from.
   - ``-f`` : File or folder used to pick out specific regions from the input image. In this case, we supply the folder of the white matter atlas that was transformed to the space of the MTR data.
   - ``-method`` : We specify ``map`` to choose the Maximum a Posteriori method, which helps to account for the partial volume effect. This method is most suitable for small regions. This method should only be used with the PAM50 white/gray matter atlas (or with any custom atlas) as long as the sum across all overlapping labels equals 1 in each voxel part of the atlas.
   - ``-l`` : The IDs of the label (or group of labels) to extract metrics for. In this case, label 53 is a combined label that represents labels 0:3 (or, the left and right fasciculus cuneatus and left and right fasciculus gracilis together). You can see the full list of labels to choose from by running ``sct_extract_metric -list-labels``.
   - ``-vert`` : This argument further restricts the metric extraction to specific vertebral levels. In this case, 3 levels in total (C2-C4) will be used for the averaging process.
   - ``-vertfile``: This argument specifies the vertebral label file to use when extracting metrics.
   - ``-o`` : The name of the output file.

.. csv-table:: ``mtr_in_dc.csv``: MTR values in dorsal columns
   :file: mtr_in_dc.csv
   :header-rows: 1

.. warning::

   Be careful to always check the associated volume fraction of your metrics (indicated by the Size [vox] column). This is especially relevant if you are restricting the metric computation to a subset of vertebral levels or axial slices; if the number of voxels included in the computation is too low, your quantifications will be unreliable. If you publish, we recommend including the volume fraction associated with all estimated metrics.
