.. _registering-multiple-images:

Tutorial 4: Registering additional images from the same session (e.g., diffusion or magnetization transfer data)
################################################################################################################

This tutorial demonstrates how to register images of another contrast alongside anatomical data from the same session. In this case, we will work with magnetization transfer data, but these steps should apply to any images that are similar in appearance to T1, T2, or T2* contrasts.

Before starting this tutorial
*****************************

1. Some of the steps in this tutorial rely on the results of a previous registration procedure, so you may need to first complete :ref:`registration-to-template`, which ensures that the following file is present:

   * ``/t2/warp_template2anat.nii.gz``: The warping field that defines the transformation from the PAM50 template to the anatomical space.

2. Open a terminal and navigate to the ``sct_course_london20/single_subject/data/mt/`` directory.

----------

Step 1: Segmenting MT1 data
***************************

First, we will run the ``sct_deepseg_sc`` command to segment the spinal cord from the image containing the magnetization transfer pulse.

.. code:: sh

   sct_deepseg_sc -i mt1.nii.gz -c t2 -qc ~/qc_singleSubj

   # Input arguments:
   #   - i: Input image
   #   - c: Contrast of the input image. T2 is chosen because of the visual similarity between MT1 and T2.
   #   - qc: Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

   # Output files/folders:
   #   - mt1_seg.nii.gz: 3D binary mask of the segmented spinal cord

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-mt-sct_deepseg_sc.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_deepseg_sc``.

----------

Step 2: Creating a mask
***********************

Next, we will create a mask to focus on the region of interest, which will increase the accuracy of the registration. Importantly, this mask is used to exclude the tissue surrounding the spinal cord, because it can move independently of the cord and negatively impact the registration.

.. code:: sh

   sct_create_mask -i mt1.nii.gz -p centerline,mt1_seg.nii.gz -size 35mm -f cylinder -o mask_mt1.nii.gz

   # Input arguments:
   #   - i: Input image.
   #   - p: Process to generate mask. By specifying 'centerline,mt1_seg.nii.gz', we tell the command to create a
   #        mask centered around the spinal cord centerline by using the segmentation file 'mt1_seg.nii.gz'
   #   - size: Size of the mask in the axial plane. (You can also specify size in pixels by omitting 'mm'.)
   #   - f: Shape of the mask.
   #   - qc: Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

   # Output files/folders:
   #   - mt1_seg.nii.gz: 3D binary mask of the segmented spinal cord

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-mt-sct_create_mask.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_create_mask``.

-----------

Step 3: Registering the PAM50 template to the MT1 space
*******************************************************

Now that we have the mask, we can transform the template image to the coordinate space of the MT1 image.

.. _mt-registraton-with-anat:

Method 1: Reusing previous registration results
===============================================

Say that you have already registered anatomical data that was acquired in the same session as your MT data. In that case, there is no need to run ``sct_register_to_template`` again, because you can reuse the warping field between the template and the anatomical space. Thus, the only part that is missing is transformation from the anatomical space to the MT space, as shown in the figure below.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/mt-registration-pipeline.png
   :align: center
   :figwidth: 65%

   ``Template->T2`` (blue) + ``T2->MT1`` (green) = ``Template->MT1`` (the desired result)

For the purpose of this tutorial, we will treat the example MT data and the example T2 data as though they were acquired in the same session. So, we will be able to reuse the ``/t2/warp_template2anat.nii.gz`` warping field generated in :ref:`registration-to-template`.

To accomplish this, we now use the ``sct_register_multimodal`` command, which is designed to co-register two images together.

.. code:: sh

   sct_register_multimodal -i $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz \
                           -d mt1.nii.gz -dseg mt1_seg.nii.gz \
                           -m mask_mt1.nii.gz -initwarp ../t2/warp_template2anat.nii.gz \
                           -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3  \
                           -owarp warp_template2mt.nii.gz -qc ~/qc_singleSubj

   # Input arguments:
   #   - i: Source image. Here, it is the PAM50 template taken from the SCT installation directory. The T2 version
   #        of the template is used due to its similarity in contrast to the MT1 data.
   #   - iseg: Segmentation corresponding to the source image. Here, it is the segmented spinal cord volume from
   #           the PAM50 template, taken from the SCT installation directory.
   #   - d: Destination image.
   #   - dseg: Segmentation corresponding to the destination image.
   #   - m: Mask image, which is used on the destination image to improve the accuracy over the region of interest.
   #   - initwarp: Initial warping field to apply to the source image. Here, we supply the 'warp_template2anat.nii.gz'
   #               file that was generated in the previous tutorial. Because we begin with the 'Template->T2'
   #               transform already applied, the warping field that is generated here will be 'Template->T2->MT1'
   #               a.k.a. 'Template->MT1'.
   #   - param: Here, we will tweak the default registration parameters to specify a different nonrigid deformation.
   #            The important change is 'algo=centermass': Because the template object is already "preregistered"
   #            from the previous tutorial (see '-initwarp'), the benefits of the default 'algo=centermassrot' have
   #            already been applied. So, we specify 'algo=centermass' in step 1 to exclude the unnecessary rotation.
   #   - owarp: The name of the output warping field. This is optional. If not supplied, the filename would be
   #            generated from the filenames '-i' and '-d', which in this case would be 'warp_PAM50_t22mt1.nii.gz'.
   #   - qc: Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

   # Output files/folders:
   #   - mt1_reg.nii.gz: TODO: Empty file. How to explain?
   #   - PAM50_t2_reg.nii.gz: The PAM50 template image, registered to the space of the MT1 image.
   #   - warp_template2mt.nii.gz: The warping field to transform the PAM50 template to the MT1 space.

.. _mt-registraton-without-anat:

Method 2: Registering MT data without anatomical images
=======================================================

In the case that you have only the MT data without the anatomical data, you can still perform registration. To do so, all you will need to do is apply the same vertebral labeling and template registration steps that were covered in :ref:`registration-to-template`.

First, we create one or two labels in the metric space. For example, if you know that your FOV is centered at C3/C4 disc, then you can create a label automatically with:

.. code:: sh

   sct_label_utils -i mt1_seg.nii.gz -create-seg-mid 4 -o label_c3c4.nii.gz

Then, you can register to the template. Note: In case the metric image has axial resolution with thick slices, we recommend to do the registration in the subject space (instead of the template space), without cord straightening.

.. code:: sh

   sct_register_to_template -i mt1.nii.gz -s mt1_seg.nii.gz -ldisc label_c3c4.nii.gz -ref subject \
                            -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=bsplinesyn,slicewise=1

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-mt-sct_register_multimodal-template.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_register_to_template`` using MT1 data.

.. important::

   Only use this method if you don't also have anatomical data. If you do have anatomical data, we recommend that you stick with :ref:`mt-registraton-with-anat`. By reusing the registration results, you ensure that you use a consistent transformation between each contrast in your analysis.

----------

Step 4: Transforming template objects into the MT1 space
********************************************************

Once we have the warping field, we can use it to warp the entire template to the MT space (including vertebral levels, WM/GM atlas, and more).

.. code:: sh

   sct_warp_template -d mt1.nii.gz -w warp_template2mt.nii.gz -a 1 -qc ~/qc_singleSubj

   # Input arguments:
   #   - d: Destination image the template will be warped to.
   #   - w: Warping field (template space to anatomical space).
   #   - a: Because '-a 1' is specified, the white and gray matter atlas will also be warped.
   #   - qc: Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

   # Output:
   #   - label/template/: This directory contains the entirety of the PAM50 template, transformed into the MT space.
   #   - label/atlas/: This direct contains 36 NIFTI volumes for WM/GM tracts, transformed into the MT space.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-mt-sct_warp_template.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_warp_template``.

----------

Next: Computing MTR for specific spinal cord regions
****************************************************

:ref:`computing-mtr-for-coregistered-mt-images` is a follow-on tutorial that uses the warped template objects to compute the MTR for specific regions of the spinal cord.