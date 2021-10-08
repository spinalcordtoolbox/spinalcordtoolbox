Fixing a failed ``sct_propseg`` segmentation
############################################

Due to contrast variations in MR imaging protocols, the contrast between the spinal cord and the cerebro-spinal fluid (CSF) can differ between MR volumes. Therefore, the propagated segmentation method may fail sometimes in presence of artifacts, low contrast, etc.

You have several options if the segmentation fails:

- Manually correct the segmentation.
- Try a different algorithm (``sct_deepseg_sc``).
- Tweak the parameters of ``sct_propseg`` to suit your data.

This page focuses on option 3 by providing some protocols to correct segmentation failures.

Detection problem
*****************

The computation of the spinal cord orientation, at each iteration of the propagation, can fail in lack of spinal cord/CSF contrast. Particularly, this situation can lead to an local over-segmentation or even to a propagation which has stopped too soon, resulting in a partial spinal cord segmentation.

Two correction protocols can be used to improve the segmentation : add centerline information and correct the image

Parameter "-init"
=================

This enables you to change the starting position of the propagation (and the detection) in the image. You can provide either an axial slice number (where 0 represents the slice furthest towards the inferior direction), or a decimal number (between 0 and 1) indicating a fraction of the image in the inferior-superior direction.

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/fixing-failed-sct_propseg-segmentations/propseg_init.png
  :width: 600

Parameter "-init-mask"
======================

Unfortunately, PropSeg is not perfect yet and can fails in detecting the spinal cord automatically. To help spinal cord detection and propagation, you can provide a binary mask (e.g. created with fslview) containing three non-null voxels at the center of the spinal cord, separated by ~1 cm in the superior-inferior direction. The middle point is the starting point of the propagation while the two other points represents the direction in which the propagation will be going. It is important to provide points that are exactly at the center of the spinal cord. Example below:

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/fixing-failed-sct_propseg-segmentations/propseg_initmask.png

For more convenience, you can also directly create the mask from an interactive viewer by typing: -init-mask viewer.
See figure below:

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/fixing-failed-sct_propseg-segmentations/propseg_viewer.png
  :width: 600

Parameter "-init-centerline"
============================

The spinal cord orientation is computed at each propagation iteration by minimizing/maximizing (depending on the contrast type) the sum of gradient magnitude at vertices positions. Bad contrast or error propagation can make orientation computation difficult.

Centerline information can be provided (using "-init-centerline" parameter) to ensure a correct orientation of the propagated deformable model. Spinal cord centerline can be a nifti image, with non-null values on centerline voxels. The orientation of the spinal cord will then be computed using a B-spline approximating the set of points extracted from this input image. You need to provide only a few points to get a proper representation of the spinal cord centerline (at least 5). The more points you provide, the better the segmentation will be. Propagation will start at the center of the centerline (this can be change using "-init" parameter) and stop at its edges. Centerline can also be provided by a text file, where each row contain x, y and z world coordinates (not pixel coordinates) of a point of the spinal cord, from the bottom to the top of the spinal cord.

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/fixing-failed-sct_propseg-segmentations/centerline_creation_3.png
  :width: 600

Segmentation problem
********************

Smoothing the image
===================

To minimize leaking problems, you could try to smooth the image along the spinal cord, and then re-run the segmentation. Here is an example of code used to generate the image below::

    sct_download_data -d sct_example_data
    cd sct_example_data/t1
    sct_propseg -i t1.nii.gz -c t1
    sct_smooth_spinalcord -i t1.nii.gz -s t1_seg.nii.gz -smooth 5
    sct_propseg -i t1_smooth.nii.gz -c t1 -init-centerline t1_seg.nii.gz

WARNING: you should ONLY use the smoothed spinal cord for segmentation. The rest of the processing (vertebral labeling, registration to template, etc.) should be done on the un-smoothed image.

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/fixing-failed-sct_propseg-segmentations/smooth_spinalcord.png
  :width: 600

Manually correcting the image
=============================

MR images can sometimes present local absence of contrast, making the spinal cord segmentation impossible. This situation can only be resolved by manually correcting the initial image. The goal is to enhance the contrast between the cord and the CSF by changing the values of some voxels. In most case you only need to modify a couple of voxels across 3-4 slices. You can use fslview to do it. More info below:

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/fixing-failed-sct_propseg-segmentations/propseg_enhance_contrast.png
  :width: 600

Parameter "-detect-radius"
==========================

In case the spinal cord is only partially segmented, you could try to act on this parameter which defines the initial diameter of the cord.

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/fixing-failed-sct_propseg-segmentations/propseg_radius.png
  :width: 600

Stretching/Compressing the image
================================

In case of a distorted cord, or a small one (e.g., mouse), you can apply an affine transformation to the image, then run the segmentation, and then compress back the segmentation.
First, create two files for compression and stretching. Example:

affine_stretch.txt::

    #Insight Transform File V1.0
    #Transform 0
    Transform: AffineTransform_double_3_3
    Parameters: 0.5 0 0 0 0.5 0 0 0 1 -X -Y -Z
    FixedParameters: 0 0
    With X, Y and Z being the physical coordinates of the center of your volume. You can get those values by opening the image on fsleyes. The green cross is automatically centered in the middle of the volume, then check the values
    "Coordinates: Scanner anatomical".

affine_compress.txt::

    #Insight Transform File V1.0
    #Transform 0
    Transform: AffineTransform_double_3_3
    Parameters: 2 0 0 0 2 0 0 0 1 0 0 0
    FixedParameters: 0 0

Then run (replace with your correct file names)::

    # stretch t2
    isct_antsApplyTransforms -d 3 -i t2.nii.gz -o t2_stretched.nii.gz -t affine_stretch.txt -r t2.nii.gz
    # run propseg
    sct_propseg -i t2_stretched.nii.gz -c t2 -radius 6
    # compress segmentation back in t2 space
    isct_antsApplyTransforms -d 3 -i t2_stretched_seg.nii.gz -o t2_stretched_seg_compressed.nii.gz -t affine_compress.txt -r t2.nii.gz
    # binarize
    sct_maths -i t2_stretched_seg_compressed.nii.gz -bin 0.5 -o t2_seg.nii.gz

Note, if you are working with compressed cord in the AP direction, then only modify the Y parameter. Example::

    Parameters: 1 0 0 0 0.7 0 0 0 1 0 0 0

Propagation problem
*******************

Parameter "-max-deformation"
============================

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/fixing-failed-sct_propseg-segmentations/propseg_max-deformation.png
  :width: 600

