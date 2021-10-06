Before starting this tutorial
#############################

1. Read through the following pages to familiarize yourself with key SCT concepts:

   * :ref:`qc`: Primer for SCT's Quality Control interface. After each step of this tutorial, you will be able to open a QC report that lets you easily evaluate the results of each command.

2. Make sure that you have the following files in your working directory:

   * ``single_subject/data/mt/mt1.nii.gz`` : A magnetization transfer image with the off-resonance RF pulse applied.
   * ``single_subject/data/mt/mt1_seg.nii.gz`` : 3D binary mask of the segmented spinal cord for ``mt1.nii.gz``.
   * ``single_subject/data/t2s/t2s.nii.gz`` : A T2* anatomical image of the spinal region.
   * ``single_subject/data/t2s/t2s_wmseg.nii.gz``: A binary mask for the white matter segmentation of the spinal cord.
   * ``single_subject/data/t2/warp_anat2template.nii.gz`` : The 4D warping field that defines the transform from a T2 anatomical image to the template image.
   * ``single_subject/data/t2/warp_template2anat.nii.gz`` : The 4D warping field that defines the inverse transform from the template image to a T2 anatomical image.

   You can get these files by downloading :sct_tutorial_data:`data_improving-registration-with-gm-seg.zip`.

.. note:: If you are :ref:`completing all of SCT's tutorials in sequence <completing-the-tutorials-in-sequence>`, your working directory should already contain the files needed for this tutorial.

3. Open a terminal and navigate to the ``single_subject/data/t2s/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2s/