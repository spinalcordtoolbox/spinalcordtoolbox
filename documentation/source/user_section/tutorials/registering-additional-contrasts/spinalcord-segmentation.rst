Spinal cord segmentation for MT1 data
#####################################

First, we will run the ``sct_deepseg_sc`` command to segment the spinal cord from the MT1 image (i.e. the image that has the RF off-resonance pulse applied).

.. code:: sh

   sct_deepseg_sc -i mt1.nii.gz -c t2 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image
   - ``-c`` : Contrast of the input image. T2 is chosen because of the visual similarity between MT1 and T2.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``mt1_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using either :ref:`Quality Control (QC) <qc>` reports or :ref:`fsleyes-instructions`.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-mt-sct_deepseg_sc.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_deepseg_sc``