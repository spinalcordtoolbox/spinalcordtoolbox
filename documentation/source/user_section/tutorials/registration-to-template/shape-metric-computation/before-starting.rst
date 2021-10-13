Before starting this tutorial
#############################

@. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2/t2_seg.nii.gz``: An 3D binary mask for the spinal cord of a T2 anatomical image.
   * ``single_subject/data/t2/label/template/PAM50_levels.nii.gz``: A PAM50 template object containing vertebral levels, that has been transformed to the space of the T2 anatomical image.

   You can get these files by downloading :sct_tutorial_data:`data_shape-metric-computation.zip`.

#. Open a terminal and navigate to the ``single_subject/data/t2/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2/