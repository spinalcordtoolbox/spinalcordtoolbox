Hands-on: Using ``sct_propseg`` on T1 data
##########################################

Next, we will switch to the T1 directory so that we can try out ``sct_propseg`` on a different contrast.

.. code:: sh

   cd ../t1
   sct_propseg -i t1.nii.gz -c t1 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image
   - ``-c`` : Contrast of the input image
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

:Output files/folders:
   - ``t1_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`Quality Control (QC) <qc>` reports. Optionally, If you have :ref:`fsleyes-instructions` installed, a ``fsleyes`` command will printed as well. You may also simply refresh the webpage generated in the previous section to see the new results.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/t1_propseg_before_after.png
  :align: center
  :figwidth: 65%

  Segmentation leakage with ``sct_propseg``

This time, however, there is an issue. The spinal cord segmentation has leaked outside of the expected area. This is caused by a bright outer region that is too close to the spinal cord. ``sct_propseg`` relies on contrast between the CSF and the spinal cord; without sufficient contrast, the segmentation may fail (as it has here).