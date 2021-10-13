.. _csa-pmj:

CSA (PMJ-based)
###############

Next, we will compute CSA based on a neurological reference, the ponto-medullary junction (PMJ), instead of the vertebral levels.
Vertebral levels give an approximation of the spinal levels, it is however imprecise and doesn’t consider neck flexion and extension `(Cadotte et al., 2015) <https://pubmed.ncbi.nlm.nih.gov/25523587/>`_.  
To overcome this limitation, CSA can be computed from a distance of a neuroanatomical reference, the PMJ. 

Briefly, the PMJ is detected and the label is added to spinal cord segmentation. The spinal cord centerline is extracted using linear interpolation and smoothing. The distance is computed along the centerline and CSA is averaged across the extent mask centered at a specified distance
`(Bedard & Cohen-Adad, 2021) <https://www.biorxiv.org/content/10.1101/2021.09.30.462636v1>`_.
In this case, we will compute CSA at 64 mm with an extent mask of 30 mm, other values can be specified following the desired region to compute CSA.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/sb/add-figures-pmj-tutorial/shape-metric-computation/csa-pmj-method.png
   :align: center

   PMJ-based CSA at 64 mm using a 30 mm extent mask.


Pontomedullary junction detection
---------------------------------
First, we proceed to the detection of the PMJ:

.. code:: sh

   sct_detect_pmj -i t2.nii.gz -c t2 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image.
   - ``-c``: Contrast of the input image.
   - ``-qc``: Directory for Quality Control reporting.
:Output files/folders:
   - ``t2_pmj.nii.gz``: An image containing the single-voxel PMJ label.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/sb/add-figures-pmj-tutorial/shape-metric-computation/io-pmj-detection.PNG
   :align: center

   Pontomedullary junction (PMJ) detection for T2.


CSA computation
---------------

Second, we compute CSA from a ditstance from the PMJ.

.. code:: sh

   sct_process_segmentation -i t2_seg.nii.gz -pmj t2_pmj.nii.gz -pmj-distance 64 -pmj-extent 30 -o csa_pmj.csv -qc ~/qc_singleSubj -qc-image t2.nii.gz

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-pmj`` : The PMJ label file.
   - ``-pmj-distance``: Distance (mm) from the PMJ to center the mask for CSA computation.
   - ``-pmj-extent``: Extent (mm) for the mask to compute and average CSA. 
   - ``-o`` : The output CSV file.
   - ``-qc``: Directory for Quality Control reporting.
   - ``-qc-image``: Input image to display in QC report. It would be the source anatomical image used to generate the spinal cord segmentation (``t2.nii.gz``).

:Output files/folders:
   - ``csa_pmj.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.

.. csv-table:: CSA values computed at 64 mm from the PMJ.
   :file: csa_pmj.csv
   :header-rows: 1
