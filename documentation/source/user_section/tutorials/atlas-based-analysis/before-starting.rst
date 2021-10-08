Before starting this tutorial
#############################

1. Read through the following pages to familiarize yourself with key SCT concepts:

   * :ref:`qc`: Primer for SCT's Quality Control interface. After each step of this tutorial, you will be able to open a QC report that lets you easily evaluate the results of each command.

2. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2s/t2s.nii.gz`` : A T2* anatomical image of the spinal region.
   * ``single_subject/data/t2s/t2s_wmseg.nii.gz``: A binary mask for the white matter segmentation of the spinal cord.
   * ``single_subject/data/t2s/t2s_gmseg.nii.gz``:  A binary mask for the gray matter segmentation of the spinal cord.
   * ``single_subject/data/mt/mt1.nii.gz``: A magnetization transfer image with the off-resonance RF pulse applied.
   * ``single_subject/data/mt/mtr.nii.gz``: An image containing the voxel-wise magnetization transfer ratio.
   * ``single_subject/data/mt/warp_template2mt.nii.gz`` : The 4D warping field that defines the transform from the template image to the MT image.

   You can get these files by downloading :sct_tutorial_data:`data_gm-wm-metric-computation.zip`.


3. Open a terminal and navigate to the ``single_subject/data/t2s/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2s/