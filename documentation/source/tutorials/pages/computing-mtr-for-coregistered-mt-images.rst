.. _computing-mtr-for-coregistered-mt-images:

Tutorial 5: Computing MTR for PAM50-registered MT data
######################################################

This tutorial will demonstrate how to generate an MTR image for MT data, then compute MTR values for specific regions of the spinal cord using the PAM50 template.


Before starting this tutorial
*****************************

1. Because this is a follow-on tutorial for :ref:`registering-multiple-images`, that tutorial must be completed beforehand, as several files are reused here.

 * ``mt1_seg.nii.gz``: The segmented spinal cord for the MT1 image (used for registering MT0 on MT1).
 * ``mask_mt1.nii.gz``: The mask surrounding the spinal cord region of interest (used for registering MT0 on MT1).
 * ``label/template``: The warped PAM50 template objects (used to compute MTR for specific regions).

2. Open a terminal and navigate to the ``sct_course_london20/single_subject/data/mt/`` directory.

----------

Step 1: Registering MT0 on MT1
******************************

Now that we have the mask, we can transform the MT0 image to the coordinate space of the MT1 image so that they are properly aligned. To do this, we use the ``sct_register_multimodal`` command, which is designed to co-register two images together.

.. code:: sh

   sct_register_multimodal -i mt0.nii.gz -d mt1.nii.gz -dseg mt1_seg.nii.gz -m mask_mt1.nii.gz \
                           -param step=1,type=im,algo=slicereg,metric=CC -x spline -qc ~/qc_singleSubj

   # Input arguments:
   #   - i: Source image.
   #   - d: Destination image.
   #   - dseg: Segmentation corresponding to the destination image.
   #   - m: Mask image, which is used on the destination image to improve the accuracy over the region of interest.
   #   - param: Here, we will tweak the default registration parameters to specify a different nonrigid deformation.
   #            In this case, only a single step is needed, because both MT images should have nearly identical shapes.
   #            Therefore, the only adjustment needed is slice-wise translation to adjust for axial shifts, which
   #            'algo=slicereg' takes care of.
   #   - x: The interpolation method used.
   #   - qc: Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

   # Output files/folders:
   #   - mt0_reg.nii.gz: The MT0 image registered to the MT1 space.
   #   - warp_mt02mt1.nii.gz: The warping field to transform the MT0 image to the MT1 space.
   #   - mt1_reg.nii.gz: The MT1 image registered to the MT0 space.
   #   - warp_mt12mt0.nii.gz: The warping field to transform the MT1 image to the MT0 space.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/mt-registration-mt0-mt1.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_register_multimodal``.

----------

Step 2: Computing MTR for the entire image
******************************************

Now that we have aligned the MT0 and MT1 images using co-registration, we can compute the magnetization transfer ratio (MTR) for each voxel.

.. code:: sh

   sct_compute_mtr -mt0 mt0_reg.nii.gz -mt1 mt1.nii.gz

   # Input arguments:
   #   - mt0: The input image without the magnetization transfer pulse.
   #   - mt1: The input image with the magnetization transfer pulse.

   # Output files/folders:
   #   - mtr.nii.gz: An image containing the voxel-wise magnetization transfer ratio.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-mt-sct_compute_mtr.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_compute_mtr``.

.. :

    Step 3: Computing MTR for specific regions
    ******************************************

    TODO: This will be filled in once the "Atlas-based analysis" section is transferred over (pp. 90-102 of the pdf ).

-----------

Extra: Computing MTSAT
**********************

SCT also offers the ``sct_compute_mtsat`` script to compute the MT saturation map and T1 map from PD-weighted, T1-weighted and MT-weighted FLASH images.

.. :

   TODO: Should an example be provided here? In the SCT course, this was only mentioned in an off-hand comment.