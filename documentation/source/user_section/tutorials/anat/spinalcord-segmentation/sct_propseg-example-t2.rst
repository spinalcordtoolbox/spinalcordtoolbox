Hands-on Example: T2
####################

Run the following command to process the image:

.. code:: sh

   sct_propseg -i t2.nii.gz -c t2 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image
   - ``-c`` : Contrast of the input image
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

:Output files/folders:
   - ``t2_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using either :ref:`Quality Control (QC) <qc>` reports or :ref:`fsleyes-instructions`.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/t2_propseg_before_after.png
  :align: center
  :figwidth: 65%
