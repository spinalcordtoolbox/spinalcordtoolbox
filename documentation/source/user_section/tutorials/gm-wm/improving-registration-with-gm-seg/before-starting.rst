Before starting this tutorial
#############################

1. Read through the following pages to familiarize yourself with key SCT concepts:

   * :ref:`qc`: Primer for SCT's Quality Control interface. After each step of this tutorial, you will be able to open a QC report that lets you easily evaluate the results of each command.

2. Make sure that you have the following files in your working directory:

   * ``mt/mt1.nii.gz`` : A magnetization transfer image with the off-resonance RF pulse applied.
   * ``mt/mt1_seg.nii.gz`` : 3D binary mask of the segmented spinal cord for ``mt1.nii.gz``.
   * ``t2s/t2s.nii.gz`` : A T2* anatomical image of the spinal region.
   * ``t2s/t2s_wmseg.nii.gz``: A binary mask for the white matter segmentation of the spinal cord.
   * ``t2/warp_anat2template.nii.gz`` : The 4D warping field that defines the transform from the anatomical image to the template image.
   * ``t2/warp_template2anat.nii.gz`` : The 4D warping field that defines the inverse transform from the template image to the T2 anatomical image.

   You can get these files by downloading :sct_tutorial_data:`data_improving-registration-with-gm-seg.zip`.

.. note:: If you are :ref:`completing all of SCT's tutorials in sequence <completing-the-tutorials-in-sequence>`, your working directory should already contain the files needed for this tutorial.

3. Open a terminal and navigate to the ``single_subject/data/t2s/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2s/