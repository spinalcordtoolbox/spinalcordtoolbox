Coregistering T1w with T2w
##########################

Now that we've preprocessed our images, we can align the T1w image with the T2w image.

To align the images, we will coregister them together. In other words, we will compute two different transformations: One to bring the T1w image into the T2w space, and one to bring the T2w image into the T1w space. To perform coregistration, we use the ``sct_register_multimodal`` command.

.. code:: sh

   sct_register_multimodal -i t1_crop.nii.gz -d ../t2/t2_crop.nii.gz \
                           -param step=1,type=im,algo=dl -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Source image.
   - ``-d`` : Destination image.
   - ``-param`` : Here, we specify ``algo=dl``, a deep learning-based registration algorithm that was trained to be agnostic to image contrasts.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``t1_crop_reg.nii.gz`` : The T1w image registered to the T2w space.
   - ``t2_crop_reg.nii.gz`` : The T2w image registered to the T1w space.
   - ``warp_t1_crop2t2_crop.nii.gz`` : The warping field to transform the T1w image to the T2w space.
   - ``warp_t2_crop2t1_crop.nii.gz`` : The warping field to transform the T2w image to the T1w space.

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`Quality Control (QC) <qc>` reports. Optionally, If you have :ref:`fsleyes-instructions` installed, a ``fsleyes`` command will printed as well.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/contrast-agnostic-registration/coregistration-t1-t2.png
   :align: center

   Input/output images for ``sct_register_multimodal``