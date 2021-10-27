Applying the gray matter segmentation algorithm
###############################################

``sct_deepseg_gm`` can be applied to T2* data using the following command:

.. code::

   sct_deepseg_gm -i t2s.nii.gz -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Destination image the template will be warped to.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output:
   - ``t2s_gmseg.nii.gz`` : The image file containing the gray matter segmentation.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/gm-wm-segmentation/io-sct_deepseg_gm.png
   :align: center
