Preprocessing steps to highlight the spinal cord
################################################

Similarly to the dMRI tutorial, prior to motion correction, it is useful to highlight the spinal cord region using a mask, as it helps to improve accuracy and speed up processing.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/processing-fmri-data/preprocessing.png
   :align: center


Computing a mean image
----------------------

First we compute the mean of the 4D fMRI data to obtain a coarse 3D approximation. This step is necessary because the relevant SCT tools are designed for individual 3D volumes, rather than 4D images.

.. code::

   sct_maths -i fmri.nii.gz -mean t -o fmri_mean.nii.gz

:Input arguments:
   - ``-i`` : The input image.
   - ``-mean`` : The dimension to compute the mean across. In this case, ``sct_maths`` will assume that the fMRI image is a 4D stack of 3D volumes, with dimension ``[x, y, z, t]``. Therefore, ``-mean t`` will average the 3D volumes across the temporal dimension.
   - ``-o``: The filename of the output image.

:Output files/folders:
   - ``fmri_mean.nii.gz`` : An 3D image containing the mean of the individual volumes in the 4D dMRI image.


Generating a spinal cord segmentation
-------------------------------------

Due to the low contrast between spinal cord and cerebrospinal fluid of fMRI data, it is difficult to directly obtain a spinal cord segmentation for fMRI data using ``sct_deepseg_sc``. So, as a workaround, we will instead obtain a spinal cord segmentation for another contrast (T2), then transform it to the space of the fMRI data.


Generating a T2 segmentation
============================

.. code::

   cd ../t2
   sct_deepseg_sc -i t2.nii.gz -c t2

:Input arguments:
   - ``-i`` : Input image
   - ``-c`` : Contrast of the input image

:Output files/folders:
   - ``t2_seg.nii.gz`` : 3D binary mask of the segmented spinal cord


Transforming the T2 segmentation to the fMRI space
==================================================

Since the segmentation image will only be used to coarsely highlight the region of interest, a complex transformation is not necessary, so we supply the ``-identity`` to ``sct_register_multimodal`` to speed up processing.

.. code::

   cd ../fmri
   sct_register_multimodal -i ../t2/t2_seg.nii.gz -d fmri_mean.nii.gz -identity 1


:Input arguments:
   - ``-i`` : Input image
   - ``-d`` : Destination image
   - ``-identity`` : Supplying this option will skip cord shape optimizations (e.g. translations, rotations, deformations) during registration. Instead, the registration will only change the properties of the input image (dimension, resolution, orientation) so that they match the properties of the destination image. (Conceptually, this is like copying the source data into the destination data without change, hence the name 'identity'.)

:Output files/folders:
   - ``t2_seg_reg.nii.gz`` : The T2 segmentation, transformed to the space of the fMRI mean image. This file is what will be used to create the spinal cord mask for fMRI motion correction.
   - ``fmri_mean_reg.nii.gz`` : The fMRI mean image, transformed to the space of the T2 segmentation.
   - ``warp_t2_seg2fmri_mean.nii.gz`` : The warping field to transform the T2 segmentation to the fMRI space.
   - ``warp_fmri_mean2t2_seg.nii.gz`` : The warping field to transform the fMRI mean image to the T2 space.

Creating a mask around the spinal cord
--------------------------------------

Now that we have a spinal cord segmentation in the space of the fMRI data, we can use it to create a mask around the fMRI spinal cord.

.. code::

   sct_create_mask -i fmri.nii.gz -p centerline,t2_seg_reg.nii.gz -size 35mm -f cylinder

:Input arguments:
   - ``-i`` : The input image to create the mask from.
   - ``-p`` : The process used to position the mask. The ``centerline`` process will compute the center of mass for each slice of ``t2_seg_reg.nii.gz``, then use those locations to center of the mask at each slice.
   - ``-f``: The shape of the mask. Here, we create cylinder around the centerline.
   - ``-size``: The {diameter? radius? TODO: Clarify in help description.} of the mask.

:Output files/folders:
   - ``mask_fmri.nii.gz`` : An imagine containing a mask surrounding the spinal cord.

.. TODO: Why is it that we use the mask crop the image for dMRI MOCO, but for fMRI MOCO, we pass the mask directly to the function? Shouldn't these two tutorials be consistent? Why don't we use `sct_dmri_moco -m` for the dMRI tutorial?