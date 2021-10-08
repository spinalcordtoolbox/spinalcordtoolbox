Before starting this tutorial
#############################

1. Read through the following pages to familiarize yourself with key SCT concepts:

   * :ref:`qc`: Primer for SCT's Quality Control interface. After each step of this tutorial, you will be able to open a QC report that lets you easily evaluate the results of each command.

2. Make sure that you have the following files in your working directory:

   * ``single_subject/data/mt/mt1.nii.gz``: A magnetization transfer image with the off-resonance RF pulse applied.
   * ``single_subject/data/mt/mtr.nii.gz``: An image containing the voxel-wise magnetization transfer ratio.
   * ``single_subject/data/mt/warp_template2mt.nii.gz`` : The 4D warping field that defines the transform from the template image to the MT image.

   You can get these files by downloading :sct_tutorial_data:`data_atlas-based-analysis.zip`.

.. note:: If you would like to learn how to compute the magnetization transfer ratio (``mtr.nii.gz``), please visit the following tutorial:

   * :ref:`mtr-computation`

   Also, if you would like to learn how to register MT data with the PAM50 template (which is how the ``warp_template2mt.nii.gz`` file was generated), please visit either of the following tutorials:

   * :ref:`registering-additional-contrasts`
   * :ref:`gm-informed-mt-registration`

3. Open a terminal and navigate to the ``single_subject/data/mt/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/mt/