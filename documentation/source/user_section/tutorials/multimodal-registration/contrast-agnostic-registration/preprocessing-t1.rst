.. _preprocessing-t1:

Preprocessing steps to highlight the spinal cord (T1w)
######################################################

Next, we repeat the exact same preprocessing steps that were completed for the T2w image.

Switching to the T1 directory
-----------------------------

Since we are now working with a different contrast, we will need to change directories.

.. code::

   cd ../t1

Generating a spinal cord segmentation
-------------------------------------

Next, we obtain a segmentation image that highlights the spinal cord.

.. code::

   sct_deepseg_sc -i t1.nii.gz -c t1 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : The input image.
   - ``-c`` : The contrast of the input image.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``t1_seg.nii.gz`` : A mask image containing the segmented spinal cord.

Creating a mask around the spinal cord
--------------------------------------

Once the segmentation is obtained, we can use it to create a mask around the cord.

.. code::

   sct_create_mask -i t1.nii.gz -p centerline,t1_seg.nii.gz -size 35mm -f cylinder -o mask_t1.nii.gz

:Input arguments:
   - ``-i`` : The input image to create the mask from.
   - ``-p`` : The process used to position the mask. The ``centerline`` process will compute the center of mass for each slice of ``t1_seg.nii.gz``, then use those locations for the center of the mask at each slice.
   - ``-f``: The shape of the mask. Here, we create cylinder around the centerline.
   - ``-size``: The diameter of the mask.

:Output files/folders:
   - ``mask_t1.nii.gz`` : An imagine containing a mask surrounding the spinal cord.

This mask will be passed to the motion correction script so that the algorithm will process only the spinal cord region.


Cropping around the spinal cord
-------------------------------

Now that we have the mask, we can use it to create a cropped image that contains the spinal cord.

.. code::

   sct_crop_image -i t1.nii.gz -m mask_t1.nii.gz

:Input arguments:
   - ``-i`` : The input image that will be cropped.
   - ``-m`` : The mask image used to select the region of interest from the input image.

:Output files/folders:
   - ``t1_crop.nii.gz`` : A cropped image containing data from ``-i`` within the region of interest from ``-m``.

This cropped image will be used during coregistration.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/contrast-agnostic-registration/preprocessing-t1.png
   :align: center

   Input/output images for ``sct_create_mask`` and ``sct_crop_image``