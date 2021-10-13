Hands-on: Using ``sct_deepseg_sc`` on T1 data
#############################################

Since we aim to improve the T1 segmentation, ensure that you are still in the T1 directory (``{PATH_TO_DOWNLOADED_DATA}/single_subject/data/t1``). Once there, run this command:

.. code:: sh

   sct_deepseg_sc -i t1.nii.gz -c t1 -ofolder deepseg -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image
   - ``-c`` : Contrast of the input image
   -  ``-ofolder`` : The folder to output files to. We specify this here so that we don't overwrite the ``t2_seg.nii.gz`` file output by ``sct_propseg``.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

:Output files/folders:
   - ``t2_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`Quality Control (QC) <qc>` reports. Optionally, If you have :ref:`fsleyes-instructions` installed, a ``fsleyes`` command will printed as well. You may also simply refresh the webpage that was generated in the previous sections to see the new results.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/t1_deepseg_before_after.png
   :align: center
   :figwidth: 65%

   No leakage with ``sct_deepseg_sc``

Looking at the relevant slices, we can see that ``sct_deepseg_sc`` has managed fix the leakage from ``sct_propseg``.