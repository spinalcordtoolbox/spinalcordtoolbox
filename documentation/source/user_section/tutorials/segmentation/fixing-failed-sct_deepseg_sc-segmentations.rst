Fixing a failed ``sct_deepseg_sc`` segmentation
###############################################

Due to contrast variations in MR imaging protocols, the contrast between the spinal cord and the cerebro-spinal fluid (CSF) can differ between MR volumes. Therefore, the segmentation method may fail sometimes in presence of artifacts, low contrast, etc.

You have several options if the segmentation fails:

- Change the kernel size from 2D to 3D using the ``-kernel`` argument.
- Change the centerline detection method using the ``-centerline`` argument.
- Try the legacy segmentation method based on mesh propagation: (``sct_propseg``)
- Check the specialized models in ``sct_deepseg`` to see if one fits your use case.
- Manually correct the segmentation.
- Ask for help on the `SCT forum <https://forum.spinalcordmri.org/c/sct/8>`_.