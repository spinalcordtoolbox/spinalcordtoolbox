Before starting this tutorial
#############################

This tutorial is intended to be run following the previous :ref:`lesion-segmentation` tutorial, as computing the lesion statistics relies on segmented lesion mask.

You can either run the :ref:`lesion-segmentation` tutorial first, or download the necessary files below:

#. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2_lesion/t2.nii.gz``: A T2w anatomical image with fake lesion (because of the difficulty to share patient data).
   * ``single_subject/data/t2_lesion/t2_sc_seg.nii.gz``: A 3D binary mask for the spinal cord of a T2 anatomical image.
   * ``single_subject/data/t2_lesion/t2_lesion_seg.nii.gz``: A image file containing a vertebral level labels.

   You can get these files by downloading `data_lesion.zip <https://github.com/spinalcordtoolbox/sct_tutorial_data/archive/refs/heads/master.zip>`_.

#. Open a terminal and navigate to the ``single_subject/data/t2_lesion/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2_lesion/
