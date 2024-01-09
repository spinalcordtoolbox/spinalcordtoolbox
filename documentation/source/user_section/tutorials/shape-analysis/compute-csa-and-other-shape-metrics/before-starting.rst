Before starting this tutorial
#############################

This tutorial is intended to be run following the previous :ref:`spinalcord-segmentation` and :ref:`vertebral-labeling` tutorials, as shape analysis relies on having a spinal cord segmentation mask as well as a vertebral level file.

You can either run those tutorials first, or download the necessary files below:

#. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2/t2_seg.nii.gz``: A 3D binary mask for the spinal cord of a T2 anatomical image.
   * ``single_subject/data/t2/t2_seg_labeled.nii.gz``: A image file containing a vertebral level labels.

   You can get these files by downloading :sct_tutorial_data:`data_shape-metric-computation.zip`.

#. Open a terminal and navigate to the ``single_subject/data/t2/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2/