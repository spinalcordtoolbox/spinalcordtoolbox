Using binary masks to compute CSA for gray and white matter
###########################################################

First, we will use the gray and white matter segmentations to compute the cross sectional area of GM and WM. This is achieved using ``sct_process_segmentation``.

.. important:: There is a limit to the precision you can achieve for a given image resolution. SCT does not truncate spurious digits when performing angle correction, so please keep in mind that there may be non-significant digits in the computed values. You may wish to compare angle-corrected values with their corresponding uncorrected values to get a sense of the limits on precision.

Compute the cord segmentation from the WM and GM segmentations (Optional)
-------------------------------------------------------------------------

In order to properly correct for the angle of the cord relative to the axial plane, we will need the full cord segmentation, which will help to estimate the centerline of the cord.

In most cases, you will already have the cord segmentation, as often the GM and WM segmentation will be derived from the spinal cord segmentation. However, in case you do not have this segmentation, you can sum the grey and white matter masks, then optionally binarize the resulting cord mask if the GM/WM masks were non-binary.

.. code::

  sct_maths -i t2s_wmseg.nii.gz -add t2s_gmseg.nii.gz -o t2s_seg.nii.gz
  sct_maths -i t2s_seg.nii.gz -bin 0.5

Compute CSA
-----------

We can now pass both the GM/WM masks and the full cord segmentation to compute the slicewise CSA:

.. code::

   sct_process_segmentation -i t2s_wmseg.nii.gz -o csa_wm.csv -perslice 1 -angle-corr-centerline t2s_seg.nii.gz

.. code::

   sct_process_segmentation -i t2s_gmseg.nii.gz -o csa_gm.csv -perslice 1 -angle-corr-centerline t2s_seg.nii.gz

:Input arguments:
   - ``-i`` : The input segmentation file.
   - ``-o`` : The output CSV file.
   - ``-perslice`` : Set this option to 1 to turn on per-slice computation.
   - ``-angle-corr-centerline``: Normally, angle correction will be applied during ``sct_process_segmentation`` to account for scans where the spinal cord is positioned at an angle with respect to the superior-inferior axis. While this works well when ``-i`` is a full spinal cord segmentation, in this case we are instead providing GM/WM segmentations to ``-i``. The irregular cross-sectional shape of these segmentations can have a negative effect on so the estimation of the cord centerline, which in turn may cause the estimated angle to be incorrect. So, here we explicitly provide a full spinal cord segmentation for angle correction purposes, to ensure accurate and consistent angle correction.
   - **Note:** Alternatively, if you don't have a full-cord segmentation, you could turn off angle correction entirely using ``-angle-corr 0``, but this will only provide accurate results if you know that your axial slices were acquired roughly orthogonal to the cord.


:Output files/folders:
   - ``csa_wm.csv`` and ``csa_gm.csv``: Two CSV files containing shape metrics for both the white and gray matter segmentations.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/gm-wm-metric-computation/io-sct_process_segmentation.png
   :align: center