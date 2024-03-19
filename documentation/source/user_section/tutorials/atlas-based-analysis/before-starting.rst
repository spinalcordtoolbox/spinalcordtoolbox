Before starting this tutorial
#############################

This tutorial demonstrates how to extract the MTR from specific tracts of the GM/WM atlas. Necessarily, then, we will re-use the results from the previous MTR and MT registration tutorials:

   * :ref:`mtr-computation`
   * :ref:`registering-additional-contrasts`
   * :ref:`gm-informed-mt-registration`

You can either run those tutorials first, or download the necessary files below:

#. Make sure that you have the following files in your working directory:

   * ``single_subject/data/mt/mt1.nii.gz``: A magnetization transfer image with the off-resonance RF pulse applied.
   * ``single_subject/data/mt/mtr.nii.gz``: An image containing the voxel-wise magnetization transfer ratio.
   * ``single_subject/data/mt/warp_template2mt.nii.gz`` : The 4D warping field that defines the transform from the template image to the MT image.

   You can get these files by downloading :sct_tutorial_data:`data_atlas-based-analysis.zip`.

#. Open a terminal and navigate to the ``single_subject/data/mt/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/mt/