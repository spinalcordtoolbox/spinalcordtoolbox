Spinal cord segmentation for MT1 data
#####################################

First, we will run the :ref:`sct_deepseg` command to segment the spinal cord from the MT1 image (i.e. the image that has the RF off-resonance pulse applied).

.. code:: sh

   sct_deepseg spinalcord -i mt1.nii.gz -qc ~/qc_singleSubj

:Input arguments:
   - ``spinalcord``: Task to perform. Here, we are using ``spinalcord`` to segment the spinal cord. This task is contrast-agnostic, meaning it can be used on any type of image (T1, T2, T2*, etc.)
   - ``-i`` : Input image.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``mt1_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`Quality Control (QC) <qc>` reports. Optionally, If you have :ref:`fsleyes-instructions` installed, a ``fsleyes`` command will printed as well.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registering-additional-contrasts/io-sct_deepseg_sc.png
   :align: center

   Input/output images for :ref:`sct_deepseg`