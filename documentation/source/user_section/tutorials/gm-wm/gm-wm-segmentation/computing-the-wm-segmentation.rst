Computing the white matter segmentation
#######################################

In the previous step, we used ``sct_deepseg_gm`` to segment the gray matter. However, to get the white matter, we essentially need the "inverse" volume. (In other words, the full spinal cord segmentation minus the gray matter segmentation.)

Segmenting the full spinal cord
-------------------------------

In order to subtract the gray matter, we will first will need to get the  full spinal cord segmentation.

.. code:: sh

   sct_deepseg_sc -i t2s.nii.gz -c t2s -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image.
   - ``-c`` : Contrast of the input image.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``t2s_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/gm-wm-segmentation/io-sct_deepseg_sc.png
   :align: center

Subtracting the gray matter
---------------------------

Now that we have the spinal cord segmentation, we can subtract the gray matter and be left with the white matter.

.. code:: sh

   sct_maths -i t2s_seg.nii.gz -sub t2s_gmseg.nii.gz -o t2s_wmseg.nii.gz

:Input arguments:
   - ``-i`` : Input image. (The full segmentation of the spinal cord.)
   - ``-sub`` : Flag to invoke the "subtract" functionality of ``sct_maths``, subtracting ``t2s_gmseg.nii.gz`` from the input image.
   - ``-o`` : The filename of the output image.

:Output files/folders:
   - ``t2s_wmseg.nii.gz`` : An image file containing the segmentation for the white matter.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/gm-wm-segmentation/io-sct_maths_gm_wm.png
   :align: center