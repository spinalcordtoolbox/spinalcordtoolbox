Coregistering T1w with T2w
##########################

Now that we've preprocessed our images, we can register the T1w image with the T2w image.

To co-register the two images together, we will compute two different transformations: One to bring the T1w image into the T2w space (forward warping field), and one to bring the T2w image into the T1w space (inverse warping field). To perform coregistration, we use the ``sct_register_multimodal`` command. Multiple algorithms are available, and here we demonstrate SCT's deep learning-based registration method, which works well for registering two images with very different contrasts. Another advantage of the deep learning method compared to ANTs is that fewer parameters need to be tweaked (i.e., it works better "out of the box").

.. code:: sh

   sct_register_multimodal -i t1_crop.nii.gz -d ../t2/t2_crop.nii.gz \
                           -param step=1,type=im,algo=dl -qc ~/qc_singleSubj -dseg ../t2/t2_seg.nii.gz

:Input arguments:
   - ``-i`` : Source image.
   - ``-d`` : Destination image.
   - ``-param`` : Here, we use very basic registration parameters as an example. (For more complex configurations, please refer to the :ref:`customizing-registration-section` section.)
      - ``step=1`` : As ``sct_register_multimodal`` can perform multi-step registration, each step is prefixed with ``step=#``. (In this example, though, we only use a single-step registration.)
      - ``type=im`` : Since this registration method only uses the input images ``-i`` and ``-d`` during registration, we specify ``type=im``, rather than ``type=seg`` or ``type=imseg``.
      - ``algo=dl`` : This corresponds to the contrast agnostic, deep learning (DL) registration method.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.
   - ``-dseg``:  Segmentation corresponding to the destination image. (Note: `-dseg` is not necessary for registration, but is provided for the `-qc` reporting to help with spinal cord visualization.)

:Output files/folders:
   - ``t1_crop_reg.nii.gz`` : The T1w image registered to the T2w space.
   - ``t2_crop_reg.nii.gz`` : The T2w image registered to the T1w space.
   - ``warp_t1_crop2t2_crop.nii.gz`` : The warping field to transform the T1w image to the T2w space.
   - ``warp_t2_crop2t1_crop.nii.gz`` : The warping field to transform the T2w image to the T1w space.

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`Quality Control (QC) <qc>` reports. Optionally, If you have :ref:`fsleyes-instructions` installed, two ``fsleyes`` commands will printed to open each of the registered images.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/contrast-agnostic-registration/coregistration-t1-t2.png
   :align: center

   Input/output images for ``sct_register_multimodal``