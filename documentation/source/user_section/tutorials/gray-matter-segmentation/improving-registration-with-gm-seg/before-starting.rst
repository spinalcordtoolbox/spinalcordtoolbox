Before starting this tutorial
#############################

This tutorial demonstrates how to use the GM/WM information obtained from the T2s contrast to improve registration for other types of contrasts.

This means that we will be re-using the files generated in previous registration tutorials, as well as the segmentation from the previous GM/WM segmentation tutorial:

- T2 files: :ref:`template-registration`
- MT files: :ref:`mtr-computation`
- T2s files: :ref:`gm-wm-segmentation`

You can either run those tutorials first, or download the necessary files below:

#. Make sure that you have the following files in your working directory:

   * T2 and T2* files (used in the initial registration to template for T2* data)

     * ``single_subject/data/t2/warp_anat2template.nii.gz`` : The 4D warping field that defines the transform from a T2 anatomical image to the template image.
     * ``single_subject/data/t2/warp_template2anat.nii.gz`` : The 4D warping field that defines the inverse transform from the template image to a T2 anatomical image.
     * ``single_subject/data/t2s/t2s.nii.gz`` : A T2* anatomical image of the spinal region.
     * ``single_subject/data/t2s/t2s_wmseg.nii.gz``: A binary mask for the white matter segmentation of the spinal cord.

   * MT files (used when improving MT registration with T2* warping fields)

     * ``single_subject/data/mt/mt1.nii.gz`` : A magnetization transfer image with the off-resonance RF pulse applied.
     * ``single_subject/data/mt/mt1_seg.nii.gz`` : 3D segmentation of the spinal cord, corresponding to the MT1 image.
     * ``single_subject/data/mt/mask_mt1.nii.gz`` :  3D binary mask surrounding the segmented spinal cord, corresponding to the MT1 image.

   You can get these files by downloading :sct_tutorial_data:`data_improving-registration-with-gm-seg.zip`.


#. Open a terminal and navigate to the ``single_subject/data/t2s/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2s/