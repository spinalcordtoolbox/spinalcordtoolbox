Before starting this tutorial
#############################

For this tutorial, because the fMRI data is axially acquired with thick slices, it helps to re-use the registration results from another contrast to initialize template registration for the dMRI data.

That is why, alongside the fMRI data, we provide T2-based warping fields below:

#. Make sure that you have the following files in your working directory:

   * ``single_subject/data/fmri/fmri.nii.gz`` : A 4D fMRI image comprised of 35 3D volumes.
   * ``single_subject/data/t2/t2.nii.gz`` : A 3D anatomical image used to create a mask around spinal cord for the fMRI image.
   * ``single_subject/data/t2/warp_template2anat.nii.gz`` : A warping field to transform the PAM50 template to the T2 space.
   * ``single_subject/data/t2/warp_anat2template.nii.gz`` : A warping field to transform the T2 image to the PAM50 template space

   You can get these files by downloading :sct_tutorial_data:`data_processing-fmri-data.zip`.


#. Open a terminal and navigate to the ``single_subject/data/fmri/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/fmri/