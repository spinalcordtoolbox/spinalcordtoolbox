.. _csa-pmj:

CSA (PMJ-based)
###############

Next, we will compute CSA based on a neurological reference, the ponto-medullary junction instead of the vertebral levels.
Vertebral levels gives an approximation of the spinal levels `(Cadotte et al., 2015)<https://pubmed.ncbi.nlm.nih.gov/25523587/>`_. It is however imprecise and doesn’t consider neck flexion and extension. 
To overcome this limitation, CSA can be computed from a distance of a neuroanatomical reference, the pontomedullary junction (PMJ). 


Pontomedullary junction detection
----------------------
First, we proceed to the detection of the PMJ:

.. code:: sh

   sct_detect_pmj -i t2.nii.gz -c t2 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image.
   - ``-c``: Contrast of the input image.
   - ``-qc``: Directory for Quality Control reporting.

# TODO: add image of the output

CSA computation
----------------------

Second, we compute CSA from a ditstance from the PMJ.

Briefly, PMJ label is added to sinal cord segmentation. The spinal cord centerline is extracted using linear interpolation and smoothing. The distance is computed along the centerline and CSA is averaged across the extent mask centered at a specified distance.
In this case, we will compute CSA at 64 mm with an extent mask of 30 mm.

.. code:: sh

   sct_process_segmentation -i t2_seg.nii.gz -pmj t2_pmj.nii.gz -pmj-distance 64 -pmj-extent 30 -o csa_pmj.csv -qc ~/qc_singleSubj -qc-image t2.nii.gz

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-pmj`` : The PMJ label file.
   - ``-pmj-distance``: Distance (mm) from the PMJ to center the mask for CSA computation.
   - ``-pmj-extent``: Extent (mm) for the mask to compute and average CSA. 
   - ``-o`` : The output CSV file.
   - ``-qc``: Directory for Quality Control reporting.
   - ``-qc-image``: Input image to display in QC report.

:Output files/folders:
   - ``csa_pmj.csv`` : A file containing the CSA values and other shape metrics. This file is partially replicated in the table below.

.. csv-table:: CSA values computed at 64 mm from the PMJ from the ponto-medullary junction
   :file: csa_pmj.csv
   :header-rows: 1