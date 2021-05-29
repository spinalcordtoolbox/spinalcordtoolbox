Contrasts
#########

Both ``sct_propseg`` and ``sct_deepseg_sc`` are designed to work with four common MRI image contrasts: T1 weighted, T2 weighted, T2* weighted, and DWI. If your image data uses a contrast not listed here, select the closest visual match to the available options. For example, FMRI images have bright cerebrospinal fluid (CSF) regions and dark spinal cord regions, so the T2 contrast option would be an appropriate choice.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/image_contrasts.png
  :align: center
  :figwidth: 75%

  Image contrasts and their corresponding ``-c`` settings