.. _lumbar-segmentation:

Using ``sct_deepseg`` to segment the lumber region of the spinal cord
#####################################################################

.. note:: Currently, ``sct_deepseg`` can only be used to segment the lumbar region of T2-weighted images. For other contrasts (T1w, T2*w, etc.), it is necessary to manually segment the lumbar region.

In this example, we begin with a full-body T2 anatomical scan (``t2.nii.gz``).

Cropping the image to highlight the lumbar region
=================================================

SCT's lumbar segmentation tool works best if the lumbar region is the central feature of the image. So, it often helps to crop out the irrelevant portions of the spinal cord (brain, cervical region, and most of the thoracic region).

In this case, the lumbar region occupies the lowermost 100 axial slices of the image. But, we want to include a bit of the thoracic region for registration purposes, so we keep an extra 100 slices corresponding to the T9-T12 region.

.. code:: sh

   sct_crop_image -i t2.nii.gz -zmax 200

:Input arguments:
   - ``-i`` : Input image
   - ``-zmax`` : The maximum z slice to keep. For an image with RPI orientation, the z axis corresponds to the axial plane, so specifying 200 means "keep the axial slices from 0-200".

:Output files/folders:
   - ``t2_crop.nii.gz`` : A cropped version of the initial input image.

Downloading the lumbar segmentation model
=========================================

In previous registration tutorials, ``sct_deepseg_sc`` was used to segment the spinal cord. However, ``sct_deepseg_sc`` works best on the cervical and thoracic regions. To segment the lumbar region, we will need to use a different tool instead (``sct_deepseg``).

``sct_deepseg`` is the more "general" cousin of ``sct_deepseg_sc``, providing many different deep learning models for segmentation beyond just the spinal cord. To view the available models, run:

.. code:: sh

   sct_deepseg -h

Then, to download and install the correct model, run:

.. code:: sh

   sct_deepseg -install-task seg_lumbar_sc_t2w

Now, ``sct_deepseg`` can be used to segment the lumbar region of the spinal cord.

Segmenting the cropped image
============================

Here, we simply feed the cropped image to the deep learning model to segment the lumbar region.

.. code:: sh

   sct_deepseg -i t2_crop.nii.gz -task seg_lumbar_sc_t2w

:Input arguments:
   - ``-i`` : Input image
   - ``-task`` : The deep learning segmentation task to apply to the image. In this case, we want `seg_lumbar_sc_t2w`.

:Output files/folders:
   - ``t2_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`Quality Control (QC) <qc>` reports.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/lumbar-registration/io_cropping-and-segmentation.png
   :align: center

   Input/output images after cropping and segmentation

