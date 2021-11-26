Before starting this tutorial
#############################

#. Make sure that you have the following files in your working directory:

   * ``single_subject/data/mt/mt1.nii.gz`` : The magnetization transfer image with the off-resonance RF pulse applied.
   * ``single_subject/data/t2/warp_template2anat.nii.gz`` : The warping field that defines the transformation from the PAM50 template to the T2 anatomical space.

   You can get these files by downloading :sct_tutorial_data:`data_coregistration.zip`.

#. Open a terminal and navigate to the ``single_subject/data/mt/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/mt/