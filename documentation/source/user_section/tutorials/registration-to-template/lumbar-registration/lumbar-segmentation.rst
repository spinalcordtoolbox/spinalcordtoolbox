.. _lumbar-segmentation:

Using ``sct_deepseg`` to segment the lumber region of the spinal cord
#####################################################################

In this example, we begin with T2 anatomical scan of the lumbar region (``t2_lumbar.nii.gz``).

SCT's lumbar segmentation tool works best if the lumbar region is the central feature of the image. So, it often helps to exclude the irrelevant portions of the spinal cord (brain, cervical region, and most of the thoracic region), as was done with this example scan.

Downloading the lumbar segmentation model
=========================================

In previous registration tutorials, :ref:`sct_deepseg spinalcord <sct_deepseg_spinalcord>` was used to segment the spinal cord. This is because ``spinalcord`` was largely trained on typical spinal cord scans spanning the cervical and thoracic regions. For some images, however, it may perform suboptimally on the lumbar region. If/when you find the ``spinalcord`` model to be insufficient, you can instead use :ref:`sct_deepseg_sc_lumbar_t2`, which was trained exclusively on lumbar scans. To check whether you have this model installed already, you may run the following:

.. code:: sh

   sct_deepseg -h

If `sc_lumbar_t2` is available and not already installed, run the following to install it:

.. code:: sh

   sct_deepseg sc_lumbar_t2 -install

Now you're ready to segment lumbar scans!

Segmenting the cropped image
============================

Here, we simply feed the cropped image to the deep learning model to segment the lumbar region.

.. code:: sh

   sct_deepseg sc_lumbar_t2 -i t2_lumbar.nii.gz -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``t2_lumbar_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/lumbar-registration/io_segmentation.png
   :align: center

   Input/output images after segmentation
