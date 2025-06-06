.. _cross-sectional-area:

Cross-sectional area (CSA)
##########################

This section demonstrates how to compute spinal cord cross-sectional area.

.. important:: There is a limit to the precision you can achieve for a given image resolution. SCT does not truncate spurious digits when performing angle correction, so please keep in mind that there may be non-significant digits in the computed values. You may wish to compare angle-corrected values with their corresponding uncorrected values (``-angle-corr 0``) to get a sense of the limits on precision.


CSA (Averaged across vertebral levels)
======================================

First, we will start by computing the cord cross-sectional area (CSA) averaged across vertebral levels. As an example, we'll choose the C3 and C4 vertebral levels, but you can specify any vertebral levels present in the vertebral level file.

.. code:: sh

   sct_process_segmentation -i t2_seg.nii.gz -vert 3:4 -vertfile t2_seg_labeled.nii.gz -o csa_c3c4.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vert`` : The vertebral levels to compute metrics across. Vertebral levels can be specified individually (``3,4``) or as a range (``3:4``).
   - ``-vertfile`` : The label file that specifies vertebral levels. Here, we use a label file generated by :ref:`sct_label_vertebrae`.
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``csa_c3c4.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.

.. csv-table:: CSA values computed for C3 and C4 vertebral levels (Averaged)
   :file: csa_c3c4.csv
   :header-rows: 1

.. _csa-perlevel:

CSA (Per level)
===============

Next, we will compute CSA for each individual vertebral level (rather than averaging).

.. code:: sh

   sct_process_segmentation -i t2_seg.nii.gz -vert 3:4 -vertfile t2_seg_labeled.nii.gz -perlevel 1 -o csa_perlevel.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-vert`` : The vertebral levels to compute metrics across. Vertebral levels can be specified individually (``3,4``) or as a range (``3:4``).
   - ``-vertfile`` : The label file that specifies vertebral levels. Here, we use a label file generated by :ref:`sct_label_vertebrae`.
   - ``-perlevel`` : Set this option to 1 to turn on per-level computation.
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``csa_perlevel.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.

.. csv-table:: CSA values computed for C3 and C4 vertebral levels
   :file: csa_perlevel.csv
   :header-rows: 1

CSA (Per axial slice)
=====================

Finally, to compute CSA for individual slices, set the ``-perslice`` argument to 1, and use ``-z`` argument to specify axial slice numbers or a range of slices. (For slice numbering, 0 represents the slice furthest towards the inferior direction.)

.. code:: sh

   sct_process_segmentation -i t2_seg.nii.gz -z 30:35 -vertfile t2_seg_labeled.nii.gz -perslice 1 -o csa_perslice.csv

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-perslice`` : Set this option to 1 to turn on per-slice computation.
   - ``-z`` : The Z-axis slices to compute metrics for. Slices can be specified individually (``30,31,32,33,34,35``) or as a range (``30:35``).
   - ``-vertfile`` : The label file that specifies vertebral levels. Even though this file is not technically necessary (given that we are specifying individual slices), it is still useful as it will identify which vertebral level each slice belongs to.
   - ``-o`` : The output CSV file.

:Output files/folders:
   - ``csa_perslice.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.

.. csv-table:: CSA values across slices 30 to 35
   :file: csa_perslice.csv
   :header-rows: 1

.. _csa-pmj:

CSA (PMJ-based)
===============

Although using vertebral levels as a reference to :ref:`compute CSA <csa-perlevel>` gives an approximation of the spinal levels, a drawback of that method is that it doesn’t consider neck flexion and extension `(Cadotte et al., 2015) <https://pubmed.ncbi.nlm.nih.gov/25523587/>`__.

To overcome this limitation, the CSA can instead be computed as a function of the distance to a neuroanatomical reference point. Here, we use the pontomedullary junction (PMJ) as a reference for computing CSA, since the distance from the PMJ along the spinal cord will vary depending on the position of the neck.

Computing the PMJ-based CSA involves a 4-step process `(Bedard & Cohen-Adad, 2022) <https://doi.org/10.3389/fnimg.2022.1031253>`__:

1. The PMJ is detected using :ref:`sct_detect_pmj`.
2. The spinal cord centerline is extracted using a segmentation of the spinal cord, then the centerline is extended to the position of the PMJ label using linear interpolation and smoothing. 
3. A mask is determined using two parameters: (1) distance along the centerline from the PMJ label, and (2) extent of the mask. 
4. The CSA is computed and averaged within this mask.

For this tutorial, we will compute CSA at a distance of 64 mm from the PMJ using a mask with a 30 mm extent. But, other values can be specified if you would like to alter the desired region to compute CSA.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/shape-metric-computation/csa-pmj-method.png
   :align: center

   PMJ-based CSA at 64 mm using a 30 mm extent mask.


PMJ detection
-------------

First, we proceed to the detection of the PMJ.

.. code:: sh

   sct_detect_pmj -i t2.nii.gz -c t2 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image.
   - ``-c``: Contrast of the input image.
   - ``-qc``: Directory for Quality Control reporting.
:Output files/folders:
   - ``t2_pmj.nii.gz``: An image containing the single-voxel PMJ label.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/shape-metric-computation/io-pmj-detection.PNG
   :align: center

   PMJ detection for T2.


CSA computation
---------------

Second, we compute CSA from a distance from the PMJ.

.. code:: sh

   sct_process_segmentation -i t2_seg.nii.gz -pmj t2_pmj.nii.gz -pmj-distance 64 -pmj-extent 30 \
                            -o csa_pmj.csv -qc ~/qc_singleSubj -qc-image t2.nii.gz

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-pmj`` : The PMJ label file.
   - ``-pmj-distance``: Distance (mm) from the PMJ to center the mask for CSA computation.
   - ``-pmj-extent``: Extent (mm) for the mask to compute and average CSA. 
   - ``-o`` : The output CSV file.
   - ``-qc``: Directory for Quality Control reporting.
   - ``-qc-image``: Image to display as the background in the QC report. Here, we supply the source anatomical image (``t2.nii.gz``) that was used to generate the spinal cord segmentation (``t2_seg.nii.gz``).

:Output files/folders:
   - ``csa_pmj.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.

.. csv-table:: CSA values computed at 64 mm from the PMJ.
   :file: csa_pmj.csv
   :header-rows: 1

.. note::

   The above commands will output the metrics in the subject space (with the original image's slice numbers) However, you can get the corresponding slice number in the PAM50 space by using the flag ``-normalize-PAM50 1``.

   .. code:: sh

      sct_process_segmentation -i t2_seg.nii.gz -vertfile t2_seg_labeled.nii.gz -perslice 1 -normalize-PAM50 1 -o csa_PAM50.csv
