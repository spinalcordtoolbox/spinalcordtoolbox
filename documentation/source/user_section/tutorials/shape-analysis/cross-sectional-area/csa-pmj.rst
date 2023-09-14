.. _csa-pmj:

CSA (PMJ-based)
###############

Although using vertebral levels as a reference to :ref:`compute CSA <csa-perlevel>` gives an approximation of the spinal levels, a drawback of that method is that it doesn’t consider neck flexion and extension `(Cadotte et al., 2015) <https://pubmed.ncbi.nlm.nih.gov/25523587/>`_.  

To overcome this limitation, the CSA can instead be computed as a function of the distance to a neuroanatomical reference point. Here, we use the pontomedullary junction (PMJ) as a reference for computing CSA, since the distance from the PMJ along the spinal cord will vary depending on the position of the neck.

Computing the PMJ-based CSA involves a 4-step process `(Bedard & Cohen-Adad, 2022) <https://doi.org/10.3389/fnimg.2022.1031253>`_: 

1. The PMJ is detected using ``sct_detect_pmj``.
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
