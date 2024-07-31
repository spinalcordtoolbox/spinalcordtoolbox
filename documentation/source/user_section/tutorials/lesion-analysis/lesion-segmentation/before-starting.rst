Before starting this tutorial
#############################

#. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2_lesion/t2_seg.nii.gz``: A 3D binary mask of the spinal cord.
   * ``single_subject/data/t2_lesion/t2_lesion.nii.gz``: A 3D binary mask of the lesion.

   You can get these files by downloading :sct_tutorial_data:`data_lesion.zip`.

.. note::

   Due to ethics, ``data_lesion.zip`` contains only spinal cord and lesion binary masks and not the T2w image used to generate them.


#. Open a terminal and navigate to the ``single_subject/data/t2_lesion/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2_lesion/