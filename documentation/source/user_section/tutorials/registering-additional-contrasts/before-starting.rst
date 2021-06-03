Before starting this tutorial
#############################

1. Make sure that you have the following files in your working directory:

   * ``single_subject/data/mt/mt1.nii.gz`` : The segmented spinal cord for the MT1 image (used for registering MT0 on MT1).
   * ``single_subject/data/t2/warp_template2anat.nii.gz`` : The warping field that defines the transformation from the PAM50 template to the T2 anatomical space.

   You can get these files by downloading :sct_tutorial_data:`data_coregistration.zip`.

.. note:: If you are :ref:`completing all of SCT's tutorials in sequence <completing-the-tutorials-in-sequence>`, your working directory should already contain the files needed for this tutorial.

2. Open a terminal and navigate to the ``/single_subject/data/mt/`` directory.