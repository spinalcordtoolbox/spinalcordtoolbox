Before starting this tutorial
#############################

1. Read through the following page to familiarize yourself with key SCT concepts:

   * :ref:`inspecting-your-results`: After some steps in this tutorial, instructions are provided to open the output images using :ref:`Quality Control (QC) <qc>` reports and :ref:`fsleyes-instructions`.

2. Make sure that you have the following files in your working directory:

   * ``single_subject/data/fmri/fmri.nii.gz`` : A 4D fMRI image comprised of 35 3D volumes.
   * ``single_subject/data/t2/t2.nii.gz`` : A 3D anatomical image used to create a mask around spinal cord for the fMRI image.
   * ``single_subject/data/t2s/warp_template2t2s.nii.gz`` : A warping field to transform the PAM50 template to the T2s space, informed by the shape of the gray matter. See "Improving registration results using white and gray matter segmentations" tutorial. (TODO: Remove this if we decide to exclude GM-informed warping fields from dMRI/fMRI tutorials.)
   * ``single_subject/data/t2s/warp_t2s2template.nii.gz`` : A warping field to transform the T2s image to the PAM50 template space, informed by the shape of the gray matter. See "Improving registration results using white and gray matter segmentations" tutorial. (TODO: Remove this if we decide to exclude GM-informed warping fields from dMRI/fMRI tutorials.)

   You can get these files by downloading :sct_tutorial_data:`data_processing-fmri-data.zip`.

.. note:: If you are :ref:`completing all of SCT's tutorials in sequence <completing-the-tutorials-in-sequence>`, your working directory should already contain the files needed for this tutorial.

3. Open a terminal and navigate to the ``single_subject/data/dmri/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/fmri/