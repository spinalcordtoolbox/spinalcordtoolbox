Hands-on Example: T1
####################

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t1_image.png
  :align: right
  :figwidth: 8%

  t1.nii.gz

Next, we will navigate to the T1 directory and verify that it contains a single T1-weighted image. If you are still in the T2 directory from the previous section, this can be done as follows:

.. code:: sh

   cd ../t1
   ls
   # Output
   # t1.nii.gz

Once here, we can run the ``sct_propseg`` command to process the image:

.. code:: sh

   sct_propseg -i t1.nii.gz -c t1 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image
   - ``-c`` : Contrast of the input image
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

:Output files/folders:
   - ``t1_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

Once the command has finished, instructions will appear at the bottom for inspecting the results using either :ref:`Quality Control (QC) <qc>` reports or :ref:`fsleyes-instructions`.

Inspecting the results using QC
*******************************

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t1_propseg_before_after.png
  :align: right
  :figwidth: 20%

  Segmentation leakage with ``sct_propseg``

As before, a Quality Control report command will be output when the script is complete. You may also simply refresh the webpage generated in the T2 section to see the new T1 results.

This time, however, there is an issue. The spinal cord segmentation has leaked outside of the expected area. This is caused by a bright outer region that is too close to the spinal cord. ``sct_propseg`` relies on contrast between the CSF and the spinal cord; without sufficient contrast, the segmentation may fail (as it has here).
