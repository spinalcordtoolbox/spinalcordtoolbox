Coregistering MT0 with MT1
##########################

Prior to computing the MTR, we will first align the MT0 image with the MT1 image. (Here, "MT0" refers to the image without the magnetization transfer pulse, and "MT1" refers to the image with the pulse.)

To align the images, we will coregister them together. In other words, we will compute two different transformations: One to bring the MT0 image into the MT1 space, and one to bring the MT1 image into the MT0 space. To perform coregistration, we use the ``sct_register_multimodal`` command.

.. code:: sh

   sct_register_multimodal -i mt0.nii.gz -d mt1.nii.gz -dseg mt1_seg.nii.gz -m mask_mt1.nii.gz \
                           -param step=1,type=im,algo=slicereg,metric=CC -x spline -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Source image.
   - ``-d`` : Destination image.
   - ``-dseg`` : Segmentation corresponding to the destination image.
   - ``-m`` : Mask image, which is used on the destination image to improve the accuracy over the region of interest.
   - ``-param`` : Here, we will tweak the default registration parameters to specify a different nonrigid deformation. In this case, only a single step is needed, because both MT images should have nearly identical shapes. Therefore, the only adjustment needed is slice-wise translation to adjust for axial shifts, which ``algo=slicereg`` takes care of.
   - ``-x`` : The interpolation method used.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``mt0_reg.nii.gz`` : The MT0 image registered to the MT1 space.
   - ``mt1_reg.nii.gz`` : The MT1 image registered to the MT0 space.
   - ``warp_mt02mt1.nii.gz`` : The warping field to transform the MT0 image to the MT1 space.
   - ``warp_mt12mt0.nii.gz`` : The warping field to transform the MT1 image to the MT0 space.

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using either :ref:`Quality Control (QC) <qc>` reports or :ref:`fsleyes-instructions`.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/mt-registration-mt0-mt1.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_register_multimodal``